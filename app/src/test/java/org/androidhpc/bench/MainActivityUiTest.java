// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.Robolectric;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.android.controller.ActivityController;
import org.robolectric.annotation.Config;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * UI smoke tests on Robolectric: the Activity renders a real report (host run on Mesa llvmpipe +
 * PoCL, checked in as a fixture) without crashing, shows the right verdict, survives recreation,
 * and every clickable view meets the 48 dp touch-target minimum. The native library does not load
 * on the JVM, so these tests cover the Java layer only. Screenshots go to build/ui-screenshots.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = 35, qualifiers = "w411dp-h891dp-xxhdpi")
public class MainActivityUiTest {

    @After public void reset() {
        BenchSession.running = false;
        BenchSession.discovering = false;
        BenchSession.reportJson = BenchSession.reportText = null;
        BenchSession.devicesJson = BenchSession.devicesText = null;
        BenchSession.reportSavedPath = null;
        BenchSession.reportElapsedMs = 0;
    }

    private static String fixture() throws Exception {
        try (InputStream in = MainActivityUiTest.class.getClassLoader().getResourceAsStream("host-report.json")) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
            return bo.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static void load(String json) {
        BenchSession.reportJson = json;
        BenchSession.reportText = "text report";
        BenchSession.devicesJson = json;
        BenchSession.devicesText = "devices";
        BenchSession.reportElapsedMs = 12_400;
        BenchSession.reportSavedPath = "/sdcard/Android/data/org.androidhpc.bench/files/reports/hpc-20261110-120000.json";
    }

    private static boolean containsText(View v, String s) {
        if (v instanceof TextView && ((TextView) v).getText().toString().contains(s)) return true;
        if (v instanceof ViewGroup) {
            ViewGroup g = (ViewGroup) v;
            for (int i = 0; i < g.getChildCount(); ++i) if (containsText(g.getChildAt(i), s)) return true;
        }
        return false;
    }

    /** Every visible clickable view must be at least 48 dp tall; the message lists all offenders. */
    private static void assertTouchTargets(View root, float density) {
        StringBuilder bad = new StringBuilder();
        collectSmallTargets(root, density, bad);
        assertTrue("touch targets under 48 dp:" + bad, bad.length() == 0);
    }

    private static void collectSmallTargets(View v, float density, StringBuilder bad) {
        if (v.getVisibility() != View.VISIBLE) return;
        if (v.isClickable() && v.getHeight() > 0 && v.getHeight() < Math.round(48 * density) - 1) {
            bad.append("\n  ").append(v.getClass().getSimpleName()).append(' ')
                    .append(v.getHeight() / density).append(" dp");
            if (v instanceof TextView) bad.append(" \"").append(((TextView) v).getText()).append('"');
            if (v.getParent() instanceof View) bad.append(" in ").append(((View) v.getParent()).getClass().getSimpleName());
        }
        if (v instanceof ViewGroup) {
            ViewGroup g = (ViewGroup) v;
            for (int i = 0; i < g.getChildCount(); ++i) collectSmallTargets(g.getChildAt(i), density, bad);
        }
    }

    private static void screenshot(MainActivity a, String name) throws Exception {
        String dir = System.getProperty("hpc.screenshotDir");
        if (dir == null) return;
        View root = a.getWindow().getDecorView().findViewById(android.R.id.content);
        ScrollView scroll = (ScrollView) ((ViewGroup) root).getChildAt(0);
        View content = scroll.getChildAt(0);
        int w = root.getWidth();
        content.measure(View.MeasureSpec.makeMeasureSpec(w, View.MeasureSpec.EXACTLY),
                View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED));
        content.layout(0, 0, w, content.getMeasuredHeight());
        Bitmap bmp = Bitmap.createBitmap(w, content.getMeasuredHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);
        c.drawColor(a.getColor(R.color.bg));
        content.draw(c);
        new File(dir).mkdirs();
        try (FileOutputStream o = new FileOutputStream(new File(dir, name + ".png"))) {
            bmp.compress(Bitmap.CompressFormat.PNG, 100, o);
        }
    }

    private static MainActivity launch() {
        ActivityController<MainActivity> c = Robolectric.buildActivity(MainActivity.class).setup();
        return c.get();
    }

    @Test public void rendersVerifiedReport() throws Exception {
        load(fixture());
        MainActivity a = launch();
        View root = a.getWindow().getDecorView();
        assertTrue(containsText(root, "All 20 results verified"));
        assertTrue(containsText(root, "SGEMM"));
        assertTrue(containsText(root, "STREAM triad"));
        assertTrue(containsText(root, "Not run: needs a verified CPU run and a verified GPU/NPU run"));
        assertTouchTargets(root, a.getResources().getDisplayMetrics().density);
        screenshot(a, "results-light");
    }

    @Test @Config(qualifiers = "w411dp-h891dp-night-xxhdpi")
    public void rendersFailuresAndSplitInDarkMode() throws Exception {
        JSONObject rep = new JSONObject(fixture());
        JSONArray rs = rep.getJSONArray("results");
        JSONObject bad = rs.getJSONObject(1);  // a test fixture mutation: one failed run
        bad.put("ran", false).put("verified", false).put("error", "prepare: vkCreateDevice failed: VK_ERROR_INITIALIZATION_FAILED");
        rep.put("all_verified", false);
        load(rep.toString());
        MainActivity a = launch();
        View root = a.getWindow().getDecorView();
        assertTrue(containsText(root, "1 of 20 results failed"));
        assertTrue(containsText(root, "VK_ERROR_INITIALIZATION_FAILED"));
        screenshot(a, "results-dark-failure");
    }

    @Test public void survivesRecreationWithoutLosingResults() throws Exception {
        load(fixture());
        ActivityController<MainActivity> c = Robolectric.buildActivity(MainActivity.class).setup();
        c.recreate();
        assertTrue(containsText(c.get().getWindow().getDecorView(), "All 20 results verified"));
    }

    @Test public void emptyStateWithoutNativeLibrary() throws Exception {
        MainActivity a = launch();
        View root = a.getWindow().getDecorView();
        assertFalse(Native.loaded);
        assertTrue(containsText(root, "native library failed to load"));
        assertTouchTargets(root, a.getResources().getDisplayMetrics().density);
        screenshot(a, "empty-no-native");
    }

    @Test public void formatting() {
        assertEquals("123", ReportRenderer.fmt(123.4).replace(',', '.'));
        assertEquals("12.3", ReportRenderer.fmt(12.34).replace(',', '.'));
        assertEquals("1.23", ReportRenderer.fmt(1.234).replace(',', '.'));
        assertEquals("512×512×512", ReportRenderer.pretty("512x512x512"));
        assertTrue(ReportRenderer.bandwidthKernel("triad"));
        assertFalse(ReportRenderer.bandwidthKernel("sgemm"));
        assertEquals("1 min 5 s", ReportRenderer.duration(65_000));
    }
}
