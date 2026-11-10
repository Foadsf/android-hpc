// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Insets;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.RelativeSizeSpan;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Single screen: pick a problem size, run, read verified results as ranked bar charts, export.
 * Built in code (no AndroidX, no XML layouts). Run state lives in {@link BenchSession} so it
 * survives Activity recreation.
 */
public class MainActivity extends Activity {
    private static final String PREFS = "hpc_ui_prefs";
    private static final int REQ_EXPORT_JSON = 41;
    private static final String[] PRESETS = {"small", "medium", "large"};
    private static final String[] PRESET_LABELS = {"Small", "Medium", "Large"};
    private static final String[] PRESET_HINTS = {"GEMM 256³", "GEMM 512³", "GEMM 1024³"};
    private static final Pattern JOB = Pattern.compile("^\\[(\\d+)/(\\d+)\\]\\s*(\\S+)\\s+(\\S+)\\s*(.*)$");

    /** The Activity currently on screen; worker threads deliver results to it. */
    private static WeakReference<MainActivity> current = new WeakReference<>(null);

    private final Handler main = new Handler(Looper.getMainLooper());
    private Ui ui;
    private ReportRenderer renderer;
    private DebugLog debugLog;
    private SharedPreferences prefs;

    private LinearLayout crashCard, quarantineCard, resultsBox, actionsCard, devicesBox, textReportBox;
    private TextView quarantineText, statusText, stepText, devicesCount, textReportView;
    private ProgressBar progress;
    private Button runBtn;
    private final Button[] presetBtns = new Button[3];
    private Switch heteroSw, bigCoresSw, zeroCopySw;
    private Button[] resultButtons = new Button[0];
    private int presetIndex;
    private boolean cancelRequested;

    private final Runnable poll = new Runnable() {
        @Override public void run() {
            if (!BenchSession.running) return;
            showProgress(Native.progress());
            main.postDelayed(this, 200);
        }
    };

    // ---- lifecycle ------------------------------------------------------------------------------

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        current = new WeakReference<>(this);
        debugLog = DebugLog.init(this);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        ui = new Ui(this);
        renderer = new ReportRenderer(ui);
        presetIndex = Math.max(0, Math.min(2, prefs.getInt("preset", 0)));
        buildUi();
        updateQuarantineUi();
        // Results already in this process (Activity recreated) are shown first, whatever happens next.
        if (BenchSession.devicesJson != null) renderDevices();
        if (BenchSession.reportJson != null) renderResults();
        setRunningUi(false);

        if (!Native.loaded) {
            String msg = "The native library failed to load, so nothing can run on this device.\n\n" + Native.loadError;
            debugLog.logLine(msg);
            statusText.setText(msg);
            statusText.setTextColor(ui.error);
            Ui.setEnabled(runBtn, false);
            if (BenchSession.devicesJson == null) {
                devicesBox.removeAllViews();
                devicesBox.addView(ui.caption("  Unavailable: native library not loaded."));
            }
            return;
        }
        Native.initLogging(debugLog.getLogFile().getAbsolutePath(), debugLog.getBreadcrumbFile().getAbsolutePath(),
                debugLog.isVerbose());
        Native.setDisabledBackends(debugLog.getDisabledFamiliesCsv());

        if (BenchSession.running) {
            setRunningUi(true);
            main.post(poll);
        } else if (BenchSession.devicesJson == null) {
            probeHardwareAsync();
        } else if (!BenchSession.discovering) {
            setReadyStatus();
        }
    }

    @Override protected void onResume() {
        super.onResume();
        current = new WeakReference<>(this);
    }

    @Override protected void onDestroy() {
        main.removeCallbacksAndMessages(null);
        if (isFinishing()) {
            if (BenchSession.running) Native.cancel();
            if (debugLog != null) debugLog.markCleanExit();
        }
        super.onDestroy();
    }

    @SuppressWarnings("deprecation")
    @Override public void onBackPressed() {
        if (!BenchSession.running) {
            super.onBackPressed();
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Benchmark is running")
                .setMessage("Leaving now stops the run after the current step and discards its results.")
                .setNegativeButton("Keep running", null)
                .setPositiveButton("Stop and leave", (d, w) -> {
                    Native.cancel();
                    finish();
                })
                .show();
    }

    // ---- UI construction ------------------------------------------------------------------------

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setClipToPadding(false);
        scroll.setBackgroundColor(ui.bg);
        final LinearLayout content = ui.vertical();
        final int side = ui.dp(16);
        content.setPadding(side, ui.dp(8), side, ui.dp(24));
        scroll.addView(content);
        // Edge-to-edge (enforced from Android 15): pad by the system bars and display cutout.
        scroll.setOnApplyWindowInsetsListener((v, insets) -> {
            int l, t, r, btm;
            if (Build.VERSION.SDK_INT >= 30) {
                Insets s = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                l = s.left; t = s.top; r = s.right; btm = s.bottom;
            } else {
                l = insets.getSystemWindowInsetLeft(); t = insets.getSystemWindowInsetTop();
                r = insets.getSystemWindowInsetRight(); btm = insets.getSystemWindowInsetBottom();
            }
            content.setPadding(side + l, ui.dp(8) + t, side + r, ui.dp(24) + btm);
            return insets;
        });

        // Keep the column readable on tablets and landscape.
        LinearLayout column = ui.vertical();
        LinearLayout.LayoutParams colLp = Ui.matchWrap();
        content.addView(column, colLp);
        content.post(() -> {
            int max = ui.dp(720);
            if (content.getWidth() > max + 2 * side) {
                colLp.width = max;
                colLp.gravity = android.view.Gravity.CENTER_HORIZONTAL;
                column.setLayoutParams(colLp);
            }
        });

        buildHeader(column);
        buildCrashCard(column);
        buildQuarantineCard(column);
        buildRunCard(column);

        resultsBox = ui.vertical();
        column.addView(resultsBox, Ui.matchWrap());
        buildActionsCard(column);
        buildDevicesSection(column);
        buildDiagnostics(column);

        setContentView(scroll);
    }

    private void buildHeader(LinearLayout col) {
        LinearLayout h = ui.vertical();
        h.setPadding(ui.dp(4), ui.dp(12), ui.dp(4), ui.dp(4));
        h.addView(ui.title("Android HPC", 26));
        String ver = "?";
        try {
            ver = getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (Exception ignored) {
        }
        TextView dev = ui.caption("v" + ver + " · " + Build.MANUFACTURER + " " + Build.MODEL + " · Android " + Build.VERSION.RELEASE);
        dev.setPadding(0, ui.dp(2), 0, 0);
        h.addView(dev);
        TextView about = ui.text("Runs five compute kernels on every CPU, GPU and NPU interface this device exposes, "
                + "and checks every result against an fp64 reference.", 14, ui.onSurfaceVariant);
        about.setPadding(0, ui.dp(8), 0, ui.dp(4));
        h.addView(about);
        col.addView(h, Ui.matchWrap());
    }

    private void buildCrashCard(LinearLayout col) {
        DebugLog.CrashReport cr = debugLog.getCrashReport();
        if (cr == null) return;
        // A last breadcrumb ending in ":begin" means a driver step never finished (a crash or kill
        // mid-step). Otherwise the app was most likely just swiped away while idle.
        final boolean midStep = cr.lastBreadcrumb != null && cr.lastBreadcrumb.endsWith(":begin");
        crashCard = ui.tintedCard(midStep ? ui.warningContainer : ui.surfaceVariant);
        TextView t = ui.title(midStep ? "The last run stopped inside a driver call" : "The last session did not exit cleanly", 16);
        t.setTextColor(midStep ? ui.warning : ui.onSurface);
        crashCard.addView(t);
        StringBuilder sb = new StringBuilder();
        if (midStep) {
            sb.append("Step: ").append(cr.lastBreadcrumb).append('\n');
            if (cr.quarantinedFamily != null && !cr.quarantinedFamily.isEmpty()) {
                sb.append("The ").append(cr.quarantinedFamily).append(" backend is skipped from now on so the app keeps working. "
                        + "Re-enable it if the crash was a one-off.");
            } else {
                sb.append("No backend was skipped.");
            }
        } else {
            sb.append("No driver step was in progress (usually the app was closed from Recents). Nothing was skipped.");
        }
        TextView body = ui.text(sb, 13.5f, ui.onSurface);
        body.setPadding(0, ui.dp(4), 0, ui.dp(8));
        crashCard.addView(body);

        final TextView tail = ui.monoBlock("Previous log: " + cr.previousLogPath + "\nLast breadcrumb: " + cr.lastBreadcrumb
                + (cr.quarantineReason == null || cr.quarantineReason.isEmpty() ? "" : "\n" + cr.quarantineReason)
                + "\n— last 40 lines —\n" + cr.tailLines);
        tail.setTextSize(10.5f);
        final HorizontalScrollView hs = new HorizontalScrollView(this);
        hs.addView(tail);
        hs.setVisibility(View.GONE);
        crashCard.addView(hs, ui.matchWrap(0, 8));

        final Button show = ui.textButton("Show log", midStep ? ui.warning : ui.primary);
        show.setOnClickListener(v -> {
            boolean vis = hs.getVisibility() == View.VISIBLE;
            hs.setVisibility(vis ? View.GONE : View.VISIBLE);
            show.setText(vis ? "Show log" : "Hide log");
        });
        Button dismiss = ui.textButton("Dismiss", midStep ? ui.warning : ui.primary);
        dismiss.setOnClickListener(v -> {
            debugLog.clearCrashReport();
            crashCard.setVisibility(View.GONE);
            updateQuarantineUi();
        });
        if (!debugLog.getDisabledFamilies().isEmpty()) {
            Button re = ui.tonalButton("Re-enable");
            re.setOnClickListener(v -> reenableAllBackends());
            crashCard.addView(ui.buttonRow(show, dismiss, re));
        } else {
            crashCard.addView(ui.buttonRow(show, dismiss));
        }
        col.addView(crashCard, ui.matchWrap(8, 4));
    }

    private void buildQuarantineCard(LinearLayout col) {
        quarantineCard = ui.tintedCard(ui.warningContainer);
        LinearLayout row = ui.horizontal();
        quarantineText = ui.text("", 13.5f, ui.warning);
        row.addView(quarantineText, Ui.weight(1f));
        Button re = ui.tonalButton("Re-enable");
        re.setOnClickListener(v -> reenableAllBackends());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.leftMargin = ui.dp(8);
        row.addView(re, lp);
        quarantineCard.addView(row);
        col.addView(quarantineCard, ui.matchWrap(8, 4));
    }

    private void buildRunCard(LinearLayout col) {
        col.addView(ui.sectionLabel("PROBLEM SIZE"));
        LinearLayout card = ui.card();
        LinearLayout seg = ui.horizontal();
        for (int i = 0; i < 3; ++i) {
            final int idx = i;
            SpannableString s = new SpannableString(PRESET_LABELS[i] + "\n" + PRESET_HINTS[i]);
            s.setSpan(new RelativeSizeSpan(0.78f), PRESET_LABELS[i].length(), s.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            Button b = ui.outlinedButton(s);
            b.setMinHeight(ui.dp(60));
            b.setMinimumHeight(ui.dp(60));
            b.setPadding(ui.dp(6), ui.dp(8), ui.dp(6), ui.dp(8));
            b.setOnClickListener(v -> selectPreset(idx));
            presetBtns[i] = b;
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f);
            if (i > 0) lp.leftMargin = ui.dp(8);
            seg.addView(b, lp);
        }
        card.addView(seg);
        selectPreset(presetIndex);

        card.addView(ui.divider(), ui.matchWrap(12, 4));
        heteroSw = ui.switchRow(card, "CPU + GPU split", "Also run one SGEMM on the CPU and the fastest GPU together",
                prefs.getBoolean("hetero", true));
        bigCoresSw = ui.switchRow(card, "Fastest CPU cores only", "Use only the big cluster on big.LITTLE CPUs",
                prefs.getBoolean("big", false));
        zeroCopySw = ui.switchRow(card, "Vulkan zero-copy", "Experimental: share inputs as AHardwareBuffer",
                prefs.getBoolean("zc", false));
        heteroSw.setOnCheckedChangeListener((v, c) -> prefs.edit().putBoolean("hetero", c).apply());
        bigCoresSw.setOnCheckedChangeListener((v, c) -> prefs.edit().putBoolean("big", c).apply());
        zeroCopySw.setOnCheckedChangeListener((v, c) -> prefs.edit().putBoolean("zc", c).apply());

        runBtn = ui.filledButton("Run benchmark");
        runBtn.setTextSize(17);
        runBtn.setMinHeight(ui.dp(56));
        runBtn.setMinimumHeight(ui.dp(56));
        runBtn.setOnClickListener(v -> {
            if (BenchSession.running) requestCancel();
            else start();
        });
        card.addView(runBtn, ui.matchWrap(12, 0));

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setProgressTintList(ColorStateList.valueOf(ui.primary));
        progress.setProgressBackgroundTintList(ColorStateList.valueOf(ui.track));
        progress.setIndeterminateTintList(ColorStateList.valueOf(ui.primary));
        progress.setMax(1000);
        progress.setVisibility(View.GONE);
        card.addView(progress, ui.matchWrap(12, 0));

        statusText = ui.text("", 14, ui.onSurface);
        statusText.setPadding(ui.dp(2), ui.dp(10), ui.dp(2), 0);
        card.addView(statusText);
        stepText = ui.caption("");
        stepText.setPadding(ui.dp(2), ui.dp(2), ui.dp(2), 0);
        stepText.setVisibility(View.GONE);
        card.addView(stepText);
        col.addView(card, Ui.matchWrap());
    }

    private void buildActionsCard(LinearLayout col) {
        actionsCard = ui.vertical();
        Button share = ui.tonalButton("Share");
        share.setOnClickListener(v -> shareReport());
        Button export = ui.tonalButton("Export JSON");
        export.setOnClickListener(v -> exportJson());
        Button copy = ui.tonalButton("Copy");
        copy.setOnClickListener(v -> copyText("android-hpc report", BenchSession.reportText, "Report copied"));
        resultButtons = new Button[]{share, export, copy};
        actionsCard.addView(ui.buttonRow(share, export, copy));
        actionsCard.setVisibility(View.GONE);
        col.addView(actionsCard, ui.matchWrap(0, 4));
    }

    private void buildDevicesSection(LinearLayout col) {
        LinearLayout head = ui.horizontal();
        head.setMinimumHeight(ui.dp(Ui.TOUCH_MIN_DP));
        TextView label = ui.sectionLabel("COMPUTE DEVICES");
        label.setPadding(ui.dp(4), 0, 0, 0);
        head.addView(label, Ui.weight(1f));
        devicesCount = ui.caption("");
        head.addView(devicesCount);
        LinearLayout.LayoutParams hlp = ui.matchWrap(10, 2);
        col.addView(head, hlp);

        LinearLayout card = ui.card();
        card.setPadding(ui.dp(8), ui.dp(6), ui.dp(8), ui.dp(6));
        devicesBox = ui.vertical();
        devicesBox.addView(ui.caption("  Detecting…"));
        card.addView(devicesBox);
        col.addView(card, Ui.matchWrap());
    }

    private void buildDiagnostics(LinearLayout col) {
        final LinearLayout body = ui.vertical();
        final Button toggle = ui.textButton("Diagnostics and logs  ▾", ui.onSurfaceVariant);
        toggle.setGravity(android.view.Gravity.START | android.view.Gravity.CENTER_VERTICAL);
        toggle.setOnClickListener(v -> {
            boolean vis = body.getVisibility() == View.VISIBLE;
            body.setVisibility(vis ? View.GONE : View.VISIBLE);
            toggle.setText(vis ? "Diagnostics and logs  ▾" : "Diagnostics and logs  ▴");
            if (!vis) refreshTextReport();
        });
        col.addView(toggle, ui.matchWrap(16, 0));

        LinearLayout card = ui.card();
        Switch verbose = ui.switchRow(card, "Verbose debug logging", "Every native step is written and synced to the log file",
                debugLog.isVerbose());
        verbose.setOnCheckedChangeListener((v, c) -> debugLog.setVerbose(c));
        card.addView(ui.divider(), ui.matchWrap(4, 8));
        card.addView(ui.caption("Log file (pull with adb, or share it):"));
        TextView path = ui.monoBlock(debugLog.getLogFile().getAbsolutePath());
        card.addView(path, ui.matchWrap(6, 8));
        Button shareLog = ui.outlinedButton("Share log");
        shareLog.setOnClickListener(v -> shareLog());
        Button copyPath = ui.outlinedButton("Copy path");
        copyPath.setOnClickListener(v -> copyText("android-hpc log path", debugLog.getLogFile().getAbsolutePath(), "Path copied"));
        card.addView(ui.buttonRow(shareLog, copyPath));

        card.addView(ui.divider(), ui.matchWrap(12, 8));
        card.addView(ui.caption("Plain-text report (as printed by hpc_cli):"));
        textReportBox = ui.vertical();
        HorizontalScrollView hs = new HorizontalScrollView(this);
        textReportView = ui.monoBlock("");
        textReportView.setTextSize(10.5f);
        textReportView.setHorizontallyScrolling(true);
        hs.addView(textReportView);
        textReportBox.addView(hs);
        card.addView(textReportBox, ui.matchWrap(6, 0));
        body.addView(card, Ui.matchWrap());
        body.setVisibility(View.GONE);
        col.addView(body, Ui.matchWrap());
    }

    // ---- state -> UI ----------------------------------------------------------------------------

    private void selectPreset(int idx) {
        if (BenchSession.running) return;
        presetIndex = idx;
        prefs.edit().putInt("preset", idx).apply();
        for (int i = 0; i < 3; ++i) {
            Button b = presetBtns[i];
            boolean sel = i == idx;
            b.setTextColor(sel ? ui.onPrimaryContainer : ui.onSurface);
            b.setBackground(ui.ripple(sel ? ui.outlined(ui.primaryContainer, ui.primary, 12) : ui.outlined(0x00000000, ui.outline, 12), 12));
            b.setSelected(sel);
            b.setContentDescription(PRESET_LABELS[i] + " problem size, " + PRESET_HINTS[i] + (sel ? ", selected" : ""));
        }
    }

    private void setRunningUi(boolean running) {
        for (Button b : presetBtns) Ui.setEnabled(b, !running);
        Ui.setEnabled(heteroSw, !running);
        Ui.setEnabled(bigCoresSw, !running);
        Ui.setEnabled(zeroCopySw, !running);
        for (Button b : resultButtons) Ui.setEnabled(b, !running);
        if (running) {
            runBtn.setText(cancelRequested ? "Stopping…" : "Cancel");
            ui.styleFilled(runBtn, ui.errorContainer, ui.error);
            Ui.setEnabled(runBtn, !cancelRequested);
            progress.setVisibility(View.VISIBLE);
            stepText.setVisibility(View.VISIBLE);
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            runBtn.setText(BenchSession.reportJson == null ? "Run benchmark" : "Run again");
            ui.styleFilled(runBtn, ui.primary, ui.onPrimary);
            Ui.setEnabled(runBtn, Native.loaded && !BenchSession.discovering);
            progress.setVisibility(View.GONE);
            stepText.setVisibility(View.GONE);
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    private void setReadyStatus() {
        statusText.setTextColor(ui.onSurfaceVariant);
        statusText.setText(BenchSession.reportJson == null
                ? "Ready. Small takes seconds; Large can take minutes and heat the phone."
                : "Ready.");
    }

    /** Parses the native progress line "[3/22] vulkan:0 sgemm 512x512x512" into a bar + text. */
    private void showProgress(String p) {
        if (p == null || p.isEmpty() || "done".equals(p) || "cancelled".equals(p) || "failed".equals(p)) return;
        Matcher m = JOB.matcher(p);
        if (m.matches()) {
            int i = Integer.parseInt(m.group(1)), n = Math.max(1, Integer.parseInt(m.group(2)));
            progress.setIndeterminate(false);
            progress.setProgress((int) (1000L * (i - 1) / n));
            statusText.setText(ReportRenderer.kernelTitle(m.group(4)) + " on " + m.group(3));
            stepText.setText("Step " + i + " of " + n + " · " + ReportRenderer.pretty(m.group(5)) + " · "
                    + ReportRenderer.duration(System.currentTimeMillis() - BenchSession.runStartedMs));
        } else {
            if (p.startsWith("Discovering")) progress.setIndeterminate(true);
            statusText.setText(p);
        }
        statusText.setTextColor(ui.onSurface);
        if (cancelRequested) statusText.setText("Stopping after the current step…");
    }

    private void updateQuarantineUi() {
        String csv = debugLog.getDisabledFamiliesCsv();
        boolean bannerShowsIt = crashCard != null && crashCard.getVisibility() == View.VISIBLE;
        if (csv.isEmpty() || bannerShowsIt) {
            quarantineCard.setVisibility(View.GONE);
        } else {
            quarantineText.setText("Skipped after an earlier crash: " + csv.replace(",", ", "));
            quarantineCard.setVisibility(View.VISIBLE);
        }
    }

    private void renderDevices() {
        devicesBox.removeAllViews();
        JSONObject d = parse(BenchSession.devicesJson);
        if (d == null || d.has("error")) {
            devicesBox.addView(ui.text(d == null ? "Device detection returned no data." : d.optString("error"), 13.5f, ui.error));
            devicesCount.setText("");
            return;
        }
        int avail = renderer.deviceRows(devicesBox, d);
        int total = d.optJSONArray("backends") == null ? 0 : d.optJSONArray("backends").length();
        devicesCount.setText(avail + " of " + total + " ready  ");
        refreshTextReport();
    }

    private void renderResults() {
        resultsBox.removeAllViews();
        JSONObject rep = parse(BenchSession.reportJson);
        if (rep == null) {
            actionsCard.setVisibility(View.GONE);
            return;
        }
        resultsBox.addView(ui.sectionLabel("RESULTS"));
        resultsBox.addView(renderer.summary(rep, BenchSession.reportElapsedMs, BenchSession.reportSavedPath), ui.matchWrap(0, 12));
        renderer.kernelCards(resultsBox, rep);
        View h = renderer.heteroCard(rep);
        if (h != null) resultsBox.addView(h, ui.matchWrap(0, 12));
        actionsCard.setVisibility(rep.has("error") ? View.GONE : View.VISIBLE);
        refreshTextReport();
    }

    private void refreshTextReport() {
        if (textReportView == null) return;
        String t = BenchSession.reportText != null ? BenchSession.reportText : BenchSession.devicesText;
        textReportView.setText(t == null ? "(nothing yet)" : t);
    }

    private static JSONObject parse(String json) {
        if (json == null || json.isEmpty()) return null;
        try {
            return new JSONObject(json);
        } catch (Exception e) {
            return null;
        }
    }

    // ---- actions --------------------------------------------------------------------------------

    private void probeHardwareAsync() {
        BenchSession.discovering = true;
        Ui.setEnabled(runBtn, false);
        statusText.setTextColor(ui.onSurfaceVariant);
        statusText.setText("Detecting compute devices…");
        progress.setIndeterminate(true);
        progress.setVisibility(View.VISIBLE);
        final boolean bigOnly = bigCoresSw.isChecked();
        final String disabledCsv = debugLog.getDisabledFamiliesCsv();
        debugLog.logLine("Starting background hardware discovery (bigOnly=" + bigOnly + ", disabled=" + disabledCsv + ")");
        new Thread(() -> {
            Native.setDisabledBackends(disabledCsv);
            final String json = Native.discover(bigOnly);
            final String text = Native.lastText();
            debugLog.logLine("Background hardware discovery finished");
            main.post(() -> {
                BenchSession.devicesJson = json;
                BenchSession.devicesText = text;
                BenchSession.discovering = false;
                MainActivity a = current.get();
                if (a != null && !a.isDestroyed()) a.onDevicesReady();
            });
        }, "hpc-discover").start();
    }

    private void onDevicesReady() {
        renderDevices();
        if (!BenchSession.running) {
            progress.setVisibility(View.GONE);
            setRunningUi(false);
            setReadyStatus();
        }
    }

    private void start() {
        if (BenchSession.running || BenchSession.discovering || !Native.loaded) return;
        BenchSession.running = true;
        BenchSession.runStartedMs = System.currentTimeMillis();
        cancelRequested = false;
        final String p = PRESETS[presetIndex];
        final int iters = "large".equals(p) ? 5 : 10;
        final boolean h = heteroSw.isChecked(), big = bigCoresSw.isChecked(), zc = zeroCopySw.isChecked();
        final String disabledCsv = debugLog.getDisabledFamiliesCsv();
        final Context app = getApplicationContext();
        debugLog.logLine("User started benchmark run: preset=" + p + " iters=" + iters + " hetero=" + h + " big=" + big
                + " zeroCopy=" + zc + " disabled=" + disabledCsv);
        resultsBox.removeAllViews();
        actionsCard.setVisibility(View.GONE);
        progress.setIndeterminate(true);
        statusText.setText("Starting…");
        stepText.setText("");
        setRunningUi(true);
        main.post(poll);
        new Thread(() -> {
            Native.setDisabledBackends(disabledCsv);
            final String json = Native.run(p, 3, iters, "", "", zc, h, big);
            final String text = Native.lastText();
            final long elapsed = System.currentTimeMillis() - BenchSession.runStartedMs;
            final String saved = autoSave(app, json);
            debugLog.logLine("Benchmark run finished in " + elapsed + " ms" + (saved != null ? "; saved " + saved : ""));
            main.post(() -> {
                BenchSession.reportJson = json;
                BenchSession.reportText = text;
                BenchSession.reportElapsedMs = elapsed;
                BenchSession.reportSavedPath = saved;
                JSONObject rep = parse(json);
                if (rep != null && rep.has("backends")) {  // the run re-probes devices; keep the newest view
                    BenchSession.devicesJson = json;
                    BenchSession.devicesText = text;
                }
                BenchSession.running = false;
                MainActivity a = current.get();
                if (a != null && !a.isDestroyed()) a.onRunFinished();
            });
        }, "hpc-run").start();
    }

    private void onRunFinished() {
        main.removeCallbacks(poll);
        cancelRequested = false;
        setRunningUi(false);
        renderDevices();
        renderResults();
        JSONObject rep = parse(BenchSession.reportJson);
        if (rep == null) {
            statusText.setTextColor(ui.error);
            statusText.setText("The run returned no report. Share the log from Diagnostics.");
        } else {
            setReadyStatus();
        }
    }

    private void requestCancel() {
        if (!BenchSession.running || cancelRequested) return;
        cancelRequested = true;
        Native.cancel();
        setRunningUi(true);
        statusText.setText("Stopping after the current step…");
    }

    private void reenableAllBackends() {
        debugLog.clearDisabledFamilies();
        if (crashCard != null) crashCard.setVisibility(View.GONE);
        updateQuarantineUi();
        Toast.makeText(this, "All backends re-enabled", Toast.LENGTH_SHORT).show();
        if (!BenchSession.running && !BenchSession.discovering && Native.loaded) probeHardwareAsync();
    }

    /** Every finished run is also written to app storage, so `adb pull` works without any tap. */
    private static String autoSave(Context app, String json) {
        if (json == null || json.isEmpty()) return null;
        try {
            File base = app.getExternalFilesDir(null);
            if (base == null) base = app.getFilesDir();
            File dir = new File(base, "reports");
            if (!dir.isDirectory() && !dir.mkdirs()) return null;
            File f = new File(dir, "hpc-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".json");
            try (FileOutputStream o = new FileOutputStream(f)) {
                o.write(json.getBytes(StandardCharsets.UTF_8));
                o.getFD().sync();
            }
            return f.getAbsolutePath();
        } catch (Exception e) {
            return null;
        }
    }

    private void exportJson() {
        if (BenchSession.reportJson == null) return;
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/json");
        i.putExtra(Intent.EXTRA_TITLE, "android-hpc-" + Build.MODEL.replaceAll("[^A-Za-z0-9_-]", "_") + "-"
                + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".json");
        try {
            startActivityForResult(i, REQ_EXPORT_JSON);
        } catch (Exception e) {
            Toast.makeText(this, "No file picker available. The report is saved at " + BenchSession.reportSavedPath,
                    Toast.LENGTH_LONG).show();
        }
    }

    @SuppressWarnings("deprecation")
    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQ_EXPORT_JSON || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        final Uri uri = data.getData();
        final String json = BenchSession.reportJson;
        final Context app = getApplicationContext();
        new Thread(() -> {
            String msg;
            try (OutputStream o = app.getContentResolver().openOutputStream(uri, "wt")) {
                if (o == null) throw new java.io.IOException("cannot open destination");
                o.write(json.getBytes(StandardCharsets.UTF_8));
                msg = "JSON exported";
            } catch (Exception e) {
                msg = "Export failed: " + e.getMessage();
                debugLog.logLine("Export JSON failed: " + e);
            }
            final String m = msg;
            main.post(() -> Toast.makeText(app, m, Toast.LENGTH_SHORT).show());
        }, "hpc-export").start();
    }

    private void shareReport() {
        if (BenchSession.reportText == null) return;
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "android-hpc results: " + Build.MANUFACTURER + " " + Build.MODEL);
        i.putExtra(Intent.EXTRA_TEXT, BenchSession.reportText);
        startActivity(Intent.createChooser(i, "Share results"));
    }

    private void shareLog() {
        String logText = "Log file: " + debugLog.getLogFile().getAbsolutePath()
                + "\nLast breadcrumb: " + (Native.loaded ? Native.lastBreadcrumb() : "n/a")
                + "\n\n" + debugLog.readCurrentLogContent();
        DebugLog.CrashReport cr = debugLog.getCrashReport();
        if (cr != null) {
            logText += "\n\n=== PREVIOUS SESSION WITHOUT CLEAN EXIT (" + cr.previousLogPath + ") ===\n"
                    + "Breadcrumb: " + cr.lastBreadcrumb + "\n" + cr.tailLines;
        }
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "android-hpc debug log");
        i.putExtra(Intent.EXTRA_TEXT, logText);
        startActivity(Intent.createChooser(i, "Share debug log"));
    }

    private void copyText(String label, String text, String toast) {
        if (text == null) return;
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (cm == null) return;
        cm.setPrimaryClip(ClipData.newPlainText(label, text));
        // Android 13+ shows its own clipboard confirmation; a toast would duplicate it.
        if (Build.VERSION.SDK_INT < 33) Toast.makeText(this, toast, Toast.LENGTH_SHORT).show();
    }
}
