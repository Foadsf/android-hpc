// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Single-screen UI: pick a preset, run, read the table, save/share the JSON or debug log. No AndroidX. */
public class MainActivity extends Activity {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private DebugLog debugLog;
    private TextView status, output, logPathView, quarantineStatusView;
    private LinearLayout crashBanner, quarantineBar;
    private Button runBtn, cancelBtn, saveBtn, shareBtn, shareLogBtn, copyPathBtn, reenableAllBtn;
    private Spinner preset;
    private CheckBox hetero, bigCores, zeroCopy, verboseCheck;
    private volatile boolean running;
    private String lastJson = "";

    private final Runnable poll = new Runnable() {
        @Override public void run() {
            if (!running) return;
            status.setText(Native.progress());
            ui.postDelayed(this, 250);
        }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        debugLog = DebugLog.init(this);
        buildUi();
        updateQuarantineUi();

        if (!Native.loaded) {
            String msg = "Native library failed to load:\n" + Native.loadError;
            debugLog.logLine(msg);
            output.setText(msg);
            runBtn.setEnabled(false);
            return;
        }

        Native.initLogging(
                debugLog.getLogFile().getAbsolutePath(),
                debugLog.getBreadcrumbFile().getAbsolutePath(),
                debugLog.isVerbose()
        );
        Native.setDisabledBackends(debugLog.getDisabledFamiliesCsv());

        probeHardwareAsync();
    }

    private void probeHardwareAsync() {
        status.setText("Probing hardware...");
        runBtn.setEnabled(false);
        final boolean bigOnly = bigCores != null && bigCores.isChecked();
        final String disabledCsv = debugLog.getDisabledFamiliesCsv();
        debugLog.logLine("Starting background hardware discovery (bigOnly=" + bigOnly
                + ", disabled=" + disabledCsv + ")");
        new Thread(() -> {
            Native.setDisabledBackends(disabledCsv);
            final String info = Native.discover(bigOnly);
            debugLog.logLine("Background hardware discovery finished");
            ui.post(() -> {
                output.setText(info);
                status.setText("Ready. Pick a preset and press Run.");
                runBtn.setEnabled(true);
            });
        }, "hpc-discover").start();
    }

    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(12), dp(16), dp(12));
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets s = insets.getInsets(WindowInsets.Type.systemBars());
                v.setPadding(dp(16) + s.left, dp(12) + s.top, dp(16) + s.right, dp(12) + s.bottom);
            } else {
                //noinspection deprecation
                v.setPadding(dp(16) + insets.getSystemWindowInsetLeft(), dp(12) + insets.getSystemWindowInsetTop(),
                        dp(16) + insets.getSystemWindowInsetRight(), dp(12) + insets.getSystemWindowInsetBottom());
            }
            return insets;
        });

        TextView title = new TextView(this);
        title.setText("Android HPC");
        title.setTextSize(20);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);
        TextView sub = new TextView(this);
        sub.setText("Heterogeneous compute benchmark: CPU, Vulkan, OpenCL, OpenGL ES, NNAPI. Every result is checked against an fp64 reference.");
        sub.setTextSize(11.5f);
        root.addView(sub);

        // Prominent log file & directory display (Requirement 2)
        logPathView = new TextView(this);
        logPathView.setTypeface(Typeface.MONOSPACE);
        logPathView.setTextSize(10f);
        logPathView.setTextIsSelectable(true);
        logPathView.setBackgroundColor(Color.parseColor("#EEF2F6"));
        logPathView.setPadding(dp(8), dp(6), dp(8), dp(6));
        logPathView.setText("Log file: " + debugLog.getLogFile().getAbsolutePath()
                + "\nLogs dir: " + debugLog.getLogsDir().getAbsolutePath());
        LinearLayout.LayoutParams lpParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lpParams.topMargin = dp(6);
        lpParams.bottomMargin = dp(4);
        root.addView(logPathView, lpParams);

        // Unclean exit crash banner (Requirement 3 & 5)
        crashBanner = new LinearLayout(this);
        crashBanner.setOrientation(LinearLayout.VERTICAL);
        crashBanner.setBackgroundColor(Color.parseColor("#FFF3CD"));
        crashBanner.setPadding(dp(8), dp(6), dp(8), dp(6));
        DebugLog.CrashReport cr = debugLog.getCrashReport();
        if (cr != null) {
            TextView cbTitle = new TextView(this);
            cbTitle.setTypeface(Typeface.DEFAULT_BOLD);
            cbTitle.setTextSize(11.5f);
            cbTitle.setText("Previous run did not exit cleanly");
            crashBanner.addView(cbTitle);

            TextView cbBody = new TextView(this);
            cbBody.setTypeface(Typeface.MONOSPACE);
            cbBody.setTextSize(9.5f);
            cbBody.setTextIsSelectable(true);
            StringBuilder sb = new StringBuilder();
            sb.append("Previous log: ").append(cr.previousLogPath).append('\n');
            sb.append("Last breadcrumb: ").append(cr.lastBreadcrumb);
            if (cr.quarantinedFamily != null && !cr.quarantinedFamily.isEmpty()) {
                sb.append('\n').append(cr.quarantineReason);
            }
            sb.append("\n--- Last 40 lines of previous log ---\n").append(cr.tailLines);
            cbBody.setText(sb.toString());

            ScrollView cbScroll = new ScrollView(this);
            cbScroll.addView(cbBody);
            crashBanner.addView(cbScroll, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, dp(110)));

            LinearLayout cbBtns = new LinearLayout(this);
            cbBtns.setOrientation(LinearLayout.HORIZONTAL);
            Button dismissBtn = new Button(this);
            dismissBtn.setText("Dismiss banner");
            dismissBtn.setAllCaps(false);
            dismissBtn.setOnClickListener(v -> {
                debugLog.clearCrashReport();
                crashBanner.setVisibility(View.GONE);
            });
            cbBtns.addView(dismissBtn);
            if (!debugLog.getDisabledFamilies().isEmpty()) {
                Button bannerReenableBtn = new Button(this);
                bannerReenableBtn.setText("Re-enable all");
                bannerReenableBtn.setAllCaps(false);
                bannerReenableBtn.setOnClickListener(v -> reenableAllBackends());
                cbBtns.addView(bannerReenableBtn);
            }
            crashBanner.addView(cbBtns);
            root.addView(crashBanner, lpParams);
        }

        // Persistent quarantine bar (shown whenever any backend family is quarantined)
        quarantineBar = new LinearLayout(this);
        quarantineBar.setOrientation(LinearLayout.HORIZONTAL);
        quarantineBar.setGravity(Gravity.CENTER_VERTICAL);
        quarantineBar.setBackgroundColor(Color.parseColor("#F8D7DA"));
        quarantineBar.setPadding(dp(8), dp(4), dp(8), dp(4));
        quarantineStatusView = new TextView(this);
        quarantineStatusView.setTextSize(11f);
        quarantineBar.addView(quarantineStatusView, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        reenableAllBtn = new Button(this);
        reenableAllBtn.setText("Re-enable all");
        reenableAllBtn.setAllCaps(false);
        reenableAllBtn.setOnClickListener(v -> reenableAllBackends());
        quarantineBar.addView(reenableAllBtn);
        root.addView(quarantineBar, lpParams);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        TextView pl = new TextView(this);
        pl.setText("Size preset: ");
        row.addView(pl);
        preset = new Spinner(this);
        preset.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                new String[]{"small", "medium", "large"}));
        preset.setSelection(0);
        row.addView(preset);
        root.addView(row);

        verboseCheck = new CheckBox(this);
        verboseCheck.setText("Verbose debug logging");
        verboseCheck.setChecked(debugLog.isVerbose());
        verboseCheck.setOnCheckedChangeListener((btn, isChecked) -> debugLog.setVerbose(isChecked));

        hetero = new CheckBox(this);
        hetero.setText("CPU + GPU split experiment");
        hetero.setChecked(true);
        bigCores = new CheckBox(this);
        bigCores.setText("CPU: fastest cores only");
        zeroCopy = new CheckBox(this);
        zeroCopy.setText("Vulkan AHardwareBuffer zero-copy (experimental)");
        root.addView(verboseCheck);
        root.addView(hetero);
        root.addView(bigCores);
        root.addView(zeroCopy);

        LinearLayout btns = new LinearLayout(this);
        btns.setOrientation(LinearLayout.HORIZONTAL);
        runBtn = new Button(this);
        runBtn.setText("Run");
        runBtn.setOnClickListener(v -> start());
        cancelBtn = new Button(this);
        cancelBtn.setText("Cancel");
        cancelBtn.setEnabled(false);
        cancelBtn.setOnClickListener(v -> {
            Native.cancel();
            status.setText("Cancelling after the current kernel...");
        });
        saveBtn = new Button(this);
        saveBtn.setText("Save JSON");
        saveBtn.setEnabled(false);
        saveBtn.setOnClickListener(v -> save());
        shareBtn = new Button(this);
        shareBtn.setText("Share");
        shareBtn.setEnabled(false);
        shareBtn.setOnClickListener(v -> share());
        for (Button x : new Button[]{runBtn, cancelBtn, saveBtn, shareBtn}) {
            x.setAllCaps(false);
            btns.addView(x, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        }
        root.addView(btns);

        // Log utility buttons (Requirement 3)
        LinearLayout logBtns = new LinearLayout(this);
        logBtns.setOrientation(LinearLayout.HORIZONTAL);
        shareLogBtn = new Button(this);
        shareLogBtn.setText("Share log");
        shareLogBtn.setAllCaps(false);
        shareLogBtn.setOnClickListener(v -> shareLog());
        copyPathBtn = new Button(this);
        copyPathBtn.setText("Copy path");
        copyPathBtn.setAllCaps(false);
        copyPathBtn.setOnClickListener(v -> copyLogPath());
        logBtns.addView(shareLogBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        logBtns.addView(copyPathBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(logBtns);

        status = new TextView(this);
        status.setTextSize(12);
        status.setPadding(0, dp(4), 0, dp(4));
        root.addView(status);

        ScrollView sv = new ScrollView(this);
        HorizontalScroll hs = new HorizontalScroll(this);
        output = new TextView(this);
        output.setTypeface(Typeface.MONOSPACE);
        output.setTextSize(10.5f);
        output.setTextIsSelectable(true);
        hs.addView(output);
        sv.addView(hs);
        root.addView(sv, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
    }

    private void updateQuarantineUi() {
        String csv = debugLog.getDisabledFamiliesCsv();
        if (csv.isEmpty()) {
            quarantineBar.setVisibility(View.GONE);
        } else {
            quarantineStatusView.setText("Quarantined backends (skipped): " + csv);
            quarantineBar.setVisibility(View.VISIBLE);
        }
    }

    private void reenableAllBackends() {
        debugLog.clearDisabledFamilies();
        updateQuarantineUi();
        Toast.makeText(this, "All backend families re-enabled", Toast.LENGTH_SHORT).show();
        if (!running && Native.loaded) {
            probeHardwareAsync();
        }
    }

    /** Horizontal scrolling for the wide results table. */
    private static final class HorizontalScroll extends android.widget.HorizontalScrollView {
        HorizontalScroll(android.content.Context c) { super(c); }
    }

    private void start() {
        if (running) return;
        running = true;
        final String p = String.valueOf(preset.getSelectedItem());
        final int iters = "large".equals(p) ? 5 : 10;
        final boolean h = hetero.isChecked(), big = bigCores.isChecked(), zc = zeroCopy.isChecked();
        final String disabledCsv = debugLog.getDisabledFamiliesCsv();
        debugLog.logLine("User started benchmark run: preset=" + p + " iters=" + iters
                + " hetero=" + h + " big=" + big + " zeroCopy=" + zc + " disabled=" + disabledCsv);
        runBtn.setEnabled(false);
        cancelBtn.setEnabled(true);
        saveBtn.setEnabled(false);
        shareBtn.setEnabled(false);
        output.setText("");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ui.post(poll);
        new Thread(() -> {
            Native.setDisabledBackends(disabledCsv);
            String json = Native.run(p, 3, iters, "", "", zc, h, big);
            final String text = Native.lastText();
            debugLog.logLine("Benchmark run finished");
            ui.post(() -> {
                running = false;
                lastJson = json;
                output.setText(text);
                status.setText(Native.progress());
                runBtn.setEnabled(true);
                cancelBtn.setEnabled(false);
                saveBtn.setEnabled(true);
                shareBtn.setEnabled(true);
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            });
        }, "hpc-run").start();
    }

    private void save() {
        try {
            File dir = getExternalFilesDir(null);
            if (dir == null) dir = getFilesDir();
            File f = new File(dir, "hpc-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".json");
            try (FileOutputStream o = new FileOutputStream(f)) {
                o.write(lastJson.getBytes(StandardCharsets.UTF_8));
                o.flush();
                o.getFD().sync();
            }
            debugLog.logLine("Saved JSON report to " + f.getAbsolutePath());
            Toast.makeText(this, "Saved " + f.getAbsolutePath(), Toast.LENGTH_LONG).show();
            status.setText("Saved " + f.getAbsolutePath() + "  (adb pull it)");
        } catch (Exception e) {
            debugLog.logLine("Save JSON failed: " + e);
            Toast.makeText(this, "Save failed: " + e, Toast.LENGTH_LONG).show();
        }
    }

    private void share() {
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "android-hpc benchmark");
        i.putExtra(Intent.EXTRA_TEXT, output.getText().toString());
        startActivity(Intent.createChooser(i, "Share results"));
    }

    private void shareLog() {
        String logText = "Log file: " + debugLog.getLogFile().getAbsolutePath()
                + "\nLast breadcrumb: " + (Native.loaded ? Native.lastBreadcrumb() : "n/a")
                + "\n\n" + debugLog.readCurrentLogContent();
        DebugLog.CrashReport cr = debugLog.getCrashReport();
        if (cr != null) {
            logText += "\n\n=== PREVIOUS UNCLEAN SESSION (" + cr.previousLogPath + ") ===\n"
                    + "Breadcrumb: " + cr.lastBreadcrumb + "\n" + cr.tailLines;
        }
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "android-hpc debug log");
        i.putExtra(Intent.EXTRA_TEXT, logText);
        startActivity(Intent.createChooser(i, "Share debug log"));
    }

    private void copyLogPath() {
        String path = debugLog.getLogFile().getAbsolutePath();
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (cm != null) {
            cm.setPrimaryClip(ClipData.newPlainText("android-hpc log path", path));
            Toast.makeText(this, "Copied: " + path, Toast.LENGTH_SHORT).show();
        }
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (running) {
            Native.cancel();
        }
        if (isFinishing() && debugLog != null) {
            debugLog.markCleanExit();
        }
    }
}
