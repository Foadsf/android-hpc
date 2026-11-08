// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.app.Activity;
import android.content.Intent;
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

/** Single-screen UI: pick a preset, run, read the table, save/share the JSON. No AndroidX. */
public class MainActivity extends Activity {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView status, output;
    private Button runBtn, cancelBtn, saveBtn, shareBtn;
    private Spinner preset;
    private CheckBox hetero, bigCores, zeroCopy;
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
        buildUi();
        if (!Native.loaded) {
            output.setText("Native library failed to load:\n" + Native.loadError);
            runBtn.setEnabled(false);
            return;
        }
        status.setText("Probing hardware...");
        new Thread(() -> {
            final String info = Native.discover(false);
            ui.post(() -> {
                output.setText(info);
                status.setText("Ready. Pick a preset and press Run.");
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
                v.setPadding(dp(16) + insets.getSystemWindowInsetLeft(), dp(12) + insets.getSystemWindowInsetTop(),
                        dp(16) + insets.getSystemWindowInsetRight(), dp(12) + insets.getSystemWindowInsetBottom());
            }
            return insets;
        });

        TextView title = new TextView(this);
        title.setText("Android HPC");
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);
        TextView sub = new TextView(this);
        sub.setText("Heterogeneous compute benchmark: CPU, Vulkan, OpenCL, OpenGL ES, NNAPI. Every result is checked against an fp64 reference.");
        sub.setTextSize(12);
        root.addView(sub);

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

        hetero = new CheckBox(this);
        hetero.setText("CPU + GPU split experiment");
        hetero.setChecked(true);
        bigCores = new CheckBox(this);
        bigCores.setText("CPU: fastest cores only");
        zeroCopy = new CheckBox(this);
        zeroCopy.setText("Vulkan AHardwareBuffer zero-copy (experimental)");
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
        cancelBtn.setOnClickListener(v -> { Native.cancel(); status.setText("Cancelling after the current kernel..."); });
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
        runBtn.setEnabled(false);
        cancelBtn.setEnabled(true);
        saveBtn.setEnabled(false);
        shareBtn.setEnabled(false);
        output.setText("");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ui.post(poll);
        new Thread(() -> {
            String json = Native.run(p, 3, iters, "", "", zc, h, big);
            final String text = Native.lastText();
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
            }
            Toast.makeText(this, "Saved " + f.getAbsolutePath(), Toast.LENGTH_LONG).show();
            status.setText("Saved " + f.getAbsolutePath() + "  (adb pull it)");
        } catch (Exception e) {
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

    @Override protected void onDestroy() {
        super.onDestroy();
        if (running) Native.cancel();
    }
}
