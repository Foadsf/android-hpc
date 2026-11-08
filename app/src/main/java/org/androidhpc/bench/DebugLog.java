// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.app.ActivityManager;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Persistent synchronous logger, session-marker crash detector, and backend quarantine manager.
 * Every line is flushed and fsynced to disk so it survives native signals or process kills.
 */
final class DebugLog {
    static final String TAG = "android-hpc";
    private static final String PREFS_NAME = "hpc_debug_prefs";
    private static final String KEY_VERBOSE = "verbose_logging";
    private static final String KEY_DISABLED_FAMILIES = "disabled_families";
    private static final int MAX_LOG_FILES = 10;

    static final class CrashReport {
        final String previousLogPath;
        final String lastBreadcrumb;
        final String quarantinedFamily;
        final String quarantineReason;
        final String tailLines;

        CrashReport(String previousLogPath, String lastBreadcrumb, String quarantinedFamily,
                    String quarantineReason, String tailLines) {
            this.previousLogPath = previousLogPath;
            this.lastBreadcrumb = lastBreadcrumb;
            this.quarantinedFamily = quarantinedFamily;
            this.quarantineReason = quarantineReason;
            this.tailLines = tailLines;
        }
    }

    private final Object lock = new Object();
    private final SharedPreferences prefs;
    private final File logsDir;
    private final File logFile;
    private final File breadcrumbFile;
    private final File sessionOpenFile;
    private FileOutputStream stream;
    private volatile boolean verbose;
    private CrashReport crashReport;

    private static DebugLog instance;
    private static boolean uncaughtHandlerInstalled;

    static synchronized DebugLog init(Context ctx) {
        if (instance != null) {
            return instance;
        }
        instance = new DebugLog(ctx.getApplicationContext());
        return instance;
    }

    private DebugLog(Context appCtx) {
        prefs = appCtx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        verbose = prefs.getBoolean(KEY_VERBOSE, true);

        File base = appCtx.getExternalFilesDir(null);
        if (base == null) {
            base = appCtx.getFilesDir();
        }
        File dir = new File(base, "logs");
        if (!dir.exists() && !dir.mkdirs()) {
            dir = new File(appCtx.getFilesDir(), "logs");
            //noinspection ResultOfMethodCallIgnored
            dir.mkdirs();
        }
        logsDir = dir;
        breadcrumbFile = new File(logsDir, "breadcrumb.txt");
        sessionOpenFile = new File(logsDir, "session.open");

        checkPreviousSessionAndQuarantine();
        pruneOldLogs(MAX_LOG_FILES - 1);

        String ts = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        logFile = new File(logsDir, "hpc-" + ts + ".log");
        try {
            stream = new FileOutputStream(logFile, true);
        } catch ( Exception e) {
            Log.e(TAG, "Failed to open log file " + logFile.getAbsolutePath(), e);
        }

        writeSessionOpenMarker();
        installUncaughtHandler();

        // Requirement 2: The absolute path of the current log file and logs directory is the
        // first line of the log and is printed to logcat (tag android-hpc).
        logLine("LOG_FILE=" + logFile.getAbsolutePath() + " LOGS_DIR=" + logsDir.getAbsolutePath());
        logStartupFacts(appCtx);
    }

    private void checkPreviousSessionAndQuarantine() {
        if (!sessionOpenFile.exists()) {
            return;
        }
        String prevLogPath = readSmallFile(sessionOpenFile).trim();
        File prevLog = prevLogPath.isEmpty() ? findNewestLogFile() : new File(prevLogPath);
        if (prevLog == null || !prevLog.exists()) {
            prevLog = findNewestLogFile();
        }
        String crumb = readSmallFile(breadcrumbFile).trim();
        if (crumb.isEmpty()) {
            crumb = "(none)";
        }
        String quarantined = "";
        String reason = "";
        String parsedFamily = parseFamilyFromBreadcrumb(crumb);
        if (parsedFamily != null && !parsedFamily.isEmpty()) {
            addDisabledFamily(parsedFamily);
            quarantined = parsedFamily;
            reason = "Previous session terminated uncleanly during '" + crumb
                    + "'. Backend family '" + parsedFamily + "' has been automatically quarantined.";
        }
        String tail = prevLog != null && prevLog.exists() ? readLastLines(prevLog, 40) : "(no previous log file found)";
        crashReport = new CrashReport(
                prevLog != null ? prevLog.getAbsolutePath() : "(unknown)",
                crumb,
                quarantined,
                reason,
                tail
        );
    }

    /**
     * Parses the breadcrumb contract:
     *   "discover:<family>:begin"
     *   "prepare:<backendId>:<kernel>:begin"
     *   "run:<backendId>:<kernel>:begin"
     * Returns the backend family if it ends in ":begin" and is one of vulkan, opencl, gles, nnapi.
     */
    static String parseFamilyFromBreadcrumb(String crumb) {
        if (crumb == null) return null;
        String c = crumb.trim();
        if (!c.endsWith(":begin")) return null;
        String[] parts = c.split(":");
        if (parts.length < 3) return null;
        String action = parts[0];
        String candidate = null;
        if ("discover".equals(action) || "prepare".equals(action) || "run".equals(action)) {
            candidate = parts[1].toLowerCase(Locale.US);
        }
        if ("vulkan".equals(candidate) || "opencl".equals(candidate)
                || "gles".equals(candidate) || "nnapi".equals(candidate)) {
            return candidate;
        }
        return null;
    }

    private void writeSessionOpenMarker() {
        try (FileOutputStream fos = new FileOutputStream(sessionOpenFile, false)) {
            byte[] data = (logFile.getAbsolutePath() + "\n").getBytes(StandardCharsets.UTF_8);
            fos.write(data);
            fos.flush();
            fos.getFD().sync();
        } catch (Exception e) {
            Log.w(TAG, "Could not write session.open marker", e);
        }
    }

    void markCleanExit() {
        logLine("Session exiting cleanly (removing session.open)");
        if (sessionOpenFile.exists()) {
            //noinspection ResultOfMethodCallIgnored
            sessionOpenFile.delete();
        }
    }

    private void installUncaughtHandler() {
        synchronized (DebugLog.class) {
            if (uncaughtHandlerInstalled) return;
            uncaughtHandlerInstalled = true;
            final Thread.UncaughtExceptionHandler prev = Thread.getDefaultUncaughtExceptionHandler();
            Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
                try {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    pw.println("=== UNCAUGHT JAVA EXCEPTION ===");
                    pw.println("Thread: " + (thread != null ? thread.getName() : "null")
                            + " (id=" + (thread != null ? thread.getId() : -1) + ")");
                    if (Native.loaded) {
                        try {
                            pw.println("Last native breadcrumb: " + Native.lastBreadcrumb());
                        } catch (Throwable ignored) {
                        }
                    }
                    if (throwable != null) {
                        throwable.printStackTrace(pw);
                    }
                    pw.println("=== END UNCAUGHT JAVA EXCEPTION ===");
                    pw.flush();
                    DebugLog dl = instance;
                    if (dl != null) {
                        dl.logRawForced(sw.toString());
                    }
                } catch (Throwable ignored) {
                } finally {
                    if (prev != null) {
                        prev.uncaughtException(thread, throwable);
                    }
                }
            });
        }
    }

    private void logStartupFacts(Context ctx) {
        String versionName = "unknown";
        long versionCode = 0;
        try {
            PackageInfo pi = ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0);
            versionName = String.valueOf(pi.versionName);
            if (Build.VERSION.SDK_INT >= 28) {
                versionCode = pi.getLongVersionCode();
            } else {
                //noinspection deprecation
                versionCode = pi.versionCode;
            }
        } catch (Exception ignored) {
        }

        String socManufacturer = "n/a";
        String socModel = "n/a";
        if (Build.VERSION.SDK_INT >= 31) {
            socManufacturer = String.valueOf(Build.SOC_MANUFACTURER);
            socModel = String.valueOf(Build.SOC_MODEL);
        }

        int memClass = -1;
        int largeMemClass = -1;
        try {
            ActivityManager am = (ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE);
            if (am != null) {
                memClass = am.getMemoryClass();
                largeMemClass = am.getLargeMemoryClass();
            }
        } catch (Exception ignored) {
        }

        DisplayMetrics dm = ctx.getResources().getDisplayMetrics();
        String abis = Build.SUPPORTED_ABIS != null ? Arrays.toString(Build.SUPPORTED_ABIS) : "[]";

        logLine("App version: " + versionName + " (code " + versionCode + ")");
        logLine("Device: MANUFACTURER=" + Build.MANUFACTURER
                + " MODEL=" + Build.MODEL
                + " DEVICE=" + Build.DEVICE
                + " HARDWARE=" + Build.HARDWARE
                + " BOARD=" + Build.BOARD);
        logLine("SoC (API 31+): SOC_MANUFACTURER=" + socManufacturer + " SOC_MODEL=" + socModel);
        logLine("Android: RELEASE=" + Build.VERSION.RELEASE + " SDK_INT=" + Build.VERSION.SDK_INT
                + " SUPPORTED_ABIS=" + abis);
        logLine("Memory class: " + memClass + " MB (large=" + largeMemClass + " MB)"
                + " DisplayMetrics=" + dm.widthPixels + "x" + dm.heightPixels + " density=" + dm.density);
        logLine("Quarantined backend families: " + getDisabledFamiliesCsv());
    }

    void logLine(String msg) {
        String ts = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(new Date());
        String line = "[" + ts + "][java:" + Thread.currentThread().getName() + "] " + msg;
        Log.i(TAG, msg);
        writeSync(line + "\n");
    }

    private void logRawForced(String block) {
        Log.e(TAG, block);
        writeSync(block.endsWith("\n") ? block : (block + "\n"));
    }

    private void writeSync(String text) {
        synchronized (lock) {
            if (stream == null) return;
            try {
                stream.write(text.getBytes(StandardCharsets.UTF_8));
                stream.flush();
                stream.getFD().sync();
            } catch (Exception e) {
                Log.e(TAG, "writeSync failed", e);
            }
        }
    }

    File getLogFile() { return logFile; }
    File getLogsDir() { return logsDir; }
    File getBreadcrumbFile() { return breadcrumbFile; }
    CrashReport getCrashReport() { return crashReport; }
    void clearCrashReport() { crashReport = null; }

    boolean isVerbose() { return verbose; }

    void setVerbose(boolean v) {
        verbose = v;
        prefs.edit().putBoolean(KEY_VERBOSE, v).apply();
        logLine("Verbose logging toggled to " + v);
        if (Native.loaded) {
            Native.setVerbose(v);
        }
    }

    Set<String> getDisabledFamilies() {
        String raw = prefs.getString(KEY_DISABLED_FAMILIES, "");
        Set<String> set = new LinkedHashSet<>();
        if (raw != null && !raw.isEmpty()) {
            for (String s : raw.split(",")) {
                String t = s.trim().toLowerCase(Locale.US);
                if (!t.isEmpty()) set.add(t);
            }
        }
        return set;
    }

    String getDisabledFamiliesCsv() {
        Set<String> s = getDisabledFamilies();
        StringBuilder sb = new StringBuilder();
        for (String item : s) {
            if (sb.length() > 0) sb.append(',');
            sb.append(item);
        }
        return sb.toString();
    }

    void addDisabledFamily(String family) {
        if (family == null || family.isEmpty()) return;
        Set<String> s = getDisabledFamilies();
        s.add(family.trim().toLowerCase(Locale.US));
        saveDisabledFamilies(s);
    }

    void clearDisabledFamilies() {
        saveDisabledFamilies(new LinkedHashSet<>());
        logLine("Cleared all quarantined backend families");
        if (Native.loaded) {
            Native.setDisabledBackends("");
        }
    }

    private void saveDisabledFamilies(Set<String> s) {
        StringBuilder sb = new StringBuilder();
        for (String item : s) {
            if (sb.length() > 0) sb.append(',');
            sb.append(item);
        }
        prefs.edit().putString(KEY_DISABLED_FAMILIES, sb.toString()).apply();
    }

    String readCurrentLogContent() {
        return readLastLines(logFile, 500);
    }

    private void pruneOldLogs(int keepCount) {
        File[] files = logsDir.listFiles((d, name) -> name.startsWith("hpc-") && name.endsWith(".log"));
        if (files == null || files.length <= keepCount) return;
        Arrays.sort(files, Comparator.comparingLong(File::lastModified));
        int toDelete = files.length - keepCount;
        for (int i = 0; i < toDelete; ++i) {
            //noinspection ResultOfMethodCallIgnored
            files[i].delete();
        }
    }

    private File findNewestLogFile() {
        File[] files = logsDir.listFiles((d, name) -> name.startsWith("hpc-") && name.endsWith(".log"));
        if (files == null || files.length == 0) return null;
        Arrays.sort(files, Comparator.comparingLong(File::lastModified));
        return files[files.length - 1];
    }

    private static String readSmallFile(File f) {
        if (f == null || !f.exists()) return "";
        try (FileInputStream fis = new FileInputStream(f)) {
            byte[] buf = new byte[(int) Math.min(f.length(), 4096)];
            int n = fis.read(buf);
            if (n <= 0) return "";
            return new String(buf, 0, n, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "";
        }
    }

    static String readLastLines(File f, int maxLines) {
        if (f == null || !f.exists()) return "";
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(f), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
                if (lines.size() > maxLines) {
                    lines.remove(0);
                }
            }
        } catch (Exception e) {
            return "(failed to read log: " + e + ")";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < lines.size(); ++i) {
            if (i > 0) sb.append('\n');
            sb.append(lines.get(i));
        }
        return sb.toString();
    }
}
