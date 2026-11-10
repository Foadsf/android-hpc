// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Turns the native JSON report into cards: a summary, one ranked bar chart per kernel, the split, devices. */
final class ReportRenderer {
    private final Ui ui;

    ReportRenderer(Ui ui) { this.ui = ui; }

    // ---- formatting ---------------------------------------------------------------------------

    static String fmt(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return "-";
        double a = Math.abs(v);
        if (a >= 100) return String.format(Locale.getDefault(), "%.0f", v);
        if (a >= 10) return String.format(Locale.getDefault(), "%.1f", v);
        return String.format(Locale.getDefault(), "%.2f", v);
    }

    static String ms(double v) {
        if (Double.isNaN(v)) return "-";
        if (v >= 100) return String.format(Locale.getDefault(), "%.0f ms", v);
        if (v >= 1) return String.format(Locale.getDefault(), "%.2f ms", v);
        return String.format(Locale.getDefault(), "%.3f ms", v);
    }

    static String sci(double v) { return String.format(Locale.US, "%.2g", v); }

    static String pretty(String size) { return size == null ? "" : size.replace('x', '×'); }

    static String duration(long ms) {
        long s = Math.round(ms / 1000.0);
        return s < 60 ? s + " s" : (s / 60) + " min " + (s % 60) + " s";
    }

    /** GB/s is meaningful for the bandwidth-bound kernels only (see README). */
    static boolean bandwidthKernel(String kernel) { return "triad".equals(kernel) || "reduce".equals(kernel); }

    static String kernelTitle(String k) {
        switch (k) {
            case "sgemm": return "SGEMM";
            case "dense": return "Dense layer";
            case "conv3x3": return "Conv 3×3";
            case "reduce": return "Reduction";
            case "triad": return "STREAM triad";
            default: return k;
        }
    }

    static String kernelBlurb(String k) {
        switch (k) {
            case "sgemm": return "Matrix multiply, compute-bound";
            case "dense": return "GEMM + bias + ReLU (one NN layer)";
            case "conv3x3": return "3×3 image filter";
            case "reduce": return "Sum of all elements, two-pass";
            case "triad": return "a + α·b, memory-bandwidth-bound";
            default: return "";
        }
    }

    // ---- summary ------------------------------------------------------------------------------

    View summary(JSONObject rep, long elapsedMs, String savedPath) {
        if (rep.has("error")) {
            LinearLayout c = ui.tintedCard(ui.errorContainer);
            c.addView(ui.title("Benchmark failed", 17));
            TextView t = ui.body(rep.optString("error"));
            t.setPadding(0, ui.dp(4), 0, 0);
            c.addView(t);
            return c;
        }
        JSONArray rs = rep.optJSONArray("results");
        int total = rs == null ? 0 : rs.length(), ok = 0;
        for (int i = 0; i < total; ++i) if (rs.optJSONObject(i).optBoolean("verified")) ++ok;
        final boolean cancelled = rep.optBoolean("cancelled");
        final boolean allOk = rep.optBoolean("all_verified") && total > 0;
        int fill, fg;
        String head;
        if (cancelled) {
            fill = ui.warningContainer; fg = ui.warning;
            head = "Cancelled · " + ok + " of " + total + " results verified";
        } else if (total == 0) {
            fill = ui.warningContainer; fg = ui.warning;
            head = "No benchmark could run on this device";
        } else if (allOk) {
            fill = ui.successContainer; fg = ui.success;
            head = "✓  All " + total + " results verified";
        } else {
            fill = ui.errorContainer; fg = ui.error;
            head = "✕  " + (total - ok) + " of " + total + " results failed";
        }
        LinearLayout c = ui.tintedCard(fill);
        TextView h = ui.title(head, 17);
        h.setTextColor(fg);
        c.addView(h);
        StringBuilder sb = new StringBuilder();
        sb.append("✓ = output matched an fp64 CPU reference · tap a row for timings · preset ").append(rep.optString("preset"))
                .append(" · ").append(rep.optInt("iterations")).append(" timed runs each");
        if (elapsedMs > 0) sb.append(" · took ").append(duration(elapsedMs));
        if (!allOk && !cancelled && total > 0) sb.append("\nFailed rows are listed in red below with the reason.");
        if (savedPath != null) sb.append("\nSaved: ").append(savedPath);
        TextView t = ui.text(sb, 13, ui.onSurface);
        t.setPadding(0, ui.dp(6), 0, 0);
        t.setTextIsSelectable(true);
        c.addView(t);
        return c;
    }

    // ---- per-kernel charts --------------------------------------------------------------------

    void kernelCards(LinearLayout parent, JSONObject rep) {
        JSONArray rs = rep.optJSONArray("results");
        if (rs == null) return;
        Map<String, List<JSONObject>> byKernel = new LinkedHashMap<>();
        for (int i = 0; i < rs.length(); ++i) {
            JSONObject r = rs.optJSONObject(i);
            if (r == null) continue;
            List<JSONObject> l = byKernel.get(r.optString("kernel"));
            if (l == null) byKernel.put(r.optString("kernel"), l = new ArrayList<>());
            l.add(r);
        }
        for (Map.Entry<String, List<JSONObject>> e : byKernel.entrySet()) {
            parent.addView(kernelCard(e.getKey(), e.getValue()), ui.matchWrap(0, 12));
        }
    }

    private double metric(JSONObject r, boolean bw) { return r.optDouble(bw ? "gbps" : "gflops", 0); }

    private View kernelCard(String kernel, List<JSONObject> rows) {
        final boolean bw = bandwidthKernel(kernel);
        // Fastest first; failures last.
        rows.sort((a, b) -> {
            boolean ga = a.optBoolean("ran") && a.optBoolean("verified"), gb = b.optBoolean("ran") && b.optBoolean("verified");
            if (ga != gb) return ga ? -1 : 1;
            return Double.compare(metric(b, bw), metric(a, bw));
        });
        double best = 0;
        for (JSONObject r : rows) if (r.optBoolean("verified")) best = Math.max(best, metric(r, bw));

        LinearLayout card = ui.card();
        LinearLayout head = ui.horizontal();
        LinearLayout titles = ui.vertical();
        titles.addView(ui.title(kernelTitle(kernel), 17));
        String size = rows.isEmpty() ? "" : pretty(rows.get(0).optString("size"));
        titles.addView(ui.caption(kernelBlurb(kernel) + " · " + size));
        head.addView(titles, Ui.weight(1f));
        head.addView(ui.chip(bw ? "GB/s" : "GFLOP/s", ui.onPrimaryContainer, ui.primaryContainer));
        card.addView(head);

        for (JSONObject r : rows) {
            card.addView(resultRow(r, bw, best), ui.matchWrap(6, 0));
        }
        return card;
    }

    private View resultRow(JSONObject r, boolean bw, double best) {
        final String id = r.optString("backend");
        final boolean ran = r.optBoolean("ran"), verified = r.optBoolean("verified");
        final double v = metric(r, bw);

        LinearLayout row = ui.vertical();
        row.setMinimumHeight(ui.dp(Ui.TOUCH_MIN_DP));
        row.setPadding(ui.dp(8), ui.dp(6), ui.dp(8), ui.dp(6));
        row.setBackground(ui.ripple(ui.rounded(0x00000000, 12), 12));

        // line 1: backend id (+ labels) ................ value
        LinearLayout line1 = ui.horizontal();
        LinearLayout idLine = ui.horizontal();
        idLine.addView(ui.title(id, 14.5f));
        if (!r.optBoolean("genuine", true)) addChip(idLine, "stand-in", ui.onSurfaceVariant, ui.surfaceVariant);
        if (r.optBoolean("zero_copy")) addChip(idLine, "zero-copy", ui.onPrimaryContainer, ui.primaryContainer);
        line1.addView(idLine, Ui.weight(1f));
        TextView val;
        if (ran && verified) {
            val = ui.title(fmt(v), 16);
        } else {
            val = ui.chip(ran ? "\u2715 mismatch" : "\u2715 failed", ui.error, ui.errorContainer);
        }
        line1.addView(val);
        row.addView(line1);

        // line 2: device name ............ "✓ 1.72 ms" (verified, median time)
        LinearLayout line2 = ui.horizontal();
        TextView dev = ui.caption(r.optString("device"));
        dev.setSingleLine(true);
        dev.setEllipsize(TextUtils.TruncateAt.END);
        line2.addView(dev, Ui.weight(1f));
        if (ran && verified) {
            TextView t = ui.text("\u2713 " + ms(timeOf(r)), 12.5f, ui.success);
            t.setPadding(ui.dp(8), 0, 0, 0);
            line2.addView(t);
        }
        row.addView(line2);

        if (ran && verified) {
            row.addView(bar(best > 0 ? v / best : 0, ui.familyColor(id)), ui.matchWrap(6, 2));
        } else {
            String err = r.optString("error");
            TextView status = ui.text(err.isEmpty() ? "No result" : err, 12.5f, ui.error);
            status.setMaxLines(3);
            status.setEllipsize(TextUtils.TruncateAt.END);
            status.setPadding(0, ui.dp(4), 0, 0);
            row.addView(status);
        }

        final TextView details = ui.monoBlock(detailText(r));
        details.setVisibility(View.GONE);
        row.addView(details, ui.matchWrap(8, 0));
        row.setOnClickListener(x -> details.setVisibility(details.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE));
        row.setContentDescription(id + ", " + (ran && verified ? fmt(v) + (bw ? " gigabytes per second" : " gigaflops") + ", verified"
                : "failed") + ". Tap for details.");
        return row;
    }

    private void addChip(LinearLayout line, String label, int fg, int bgColor) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.leftMargin = ui.dp(6);
        TextView chip = ui.chip(label, fg, bgColor);
        chip.setTextSize(11);
        line.addView(chip, lp);
    }

    private static double timeOf(JSONObject r) {
        JSONObject s = r.optJSONObject("device".equals(r.optString("time_basis")) ? "device_ms" : "wall_ms");
        return s == null ? Double.NaN : s.optDouble("median", Double.NaN);
    }

    private View bar(double fraction, int color) {
        fraction = Math.max(0, Math.min(1, fraction));
        LinearLayout track = ui.horizontal();
        track.setBackground(ui.rounded(ui.track, 999));
        View fill = new View(ui.ctx);
        fill.setBackground(ui.rounded(color, 999));
        float f = fraction > 0 ? (float) Math.max(fraction, 0.015) : 0f;
        track.addView(fill, new LinearLayout.LayoutParams(0, ui.dp(8), f));
        View rest = new View(ui.ctx);
        track.addView(rest, new LinearLayout.LayoutParams(0, ui.dp(8), 1f - f));
        track.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        return track;
    }

    private static String detailText(JSONObject r) {
        StringBuilder sb = new StringBuilder();
        JSONObject w = r.optJSONObject("wall_ms"), d = r.optJSONObject("device_ms");
        sb.append("size        ").append(r.optString("size")).append('\n');
        if (w != null && w.optInt("count") > 0) {
            sb.append("wall ms     median ").append(sci3(w, "median")).append("  min ").append(sci3(w, "min"))
                    .append("\n            p95 ").append(sci3(w, "p95")).append("  sd ").append(sci3(w, "stddev"))
                    .append("  n=").append(w.optInt("count")).append('\n');
        }
        if (d != null && d.optInt("count") > 0) {
            sb.append("device ms   median ").append(sci3(d, "median")).append("  p95 ").append(sci3(d, "p95")).append('\n');
        } else if (r.optInt("device_samples_rejected") > 0) {
            sb.append("device ms   discarded: timer inconsistent with wall time\n");
        } else {
            sb.append("device ms   not reported by this API\n");
        }
        sb.append("time basis  ").append("device".equals(r.optString("time_basis")) ? "device timer (API-reported)" : "wall clock (host)").append('\n');
        sb.append("GFLOP/s     ").append(fmt(r.optDouble("gflops", 0))).append("   GB/s ").append(fmt(r.optDouble("gbps", 0))).append('\n');
        sb.append("prepare     ").append(ms(r.optDouble("prepare_ms", Double.NaN))).append(" (not timed)\n");
        sb.append("max |err|   ").append(sci(r.optDouble("max_abs_err", 0)))
                .append("   rel ").append(sci(r.optDouble("max_rel_err", 0)));
        String err = r.optString("error");
        if (!err.isEmpty()) sb.append("\nerror       ").append(err);
        return sb.toString();
    }

    private static String sci3(JSONObject s, String key) {
        double v = s.optDouble(key, Double.NaN);
        return Double.isNaN(v) ? "-" : String.format(Locale.US, "%.3f", v);
    }

    // ---- CPU + GPU split ----------------------------------------------------------------------

    View heteroCard(JSONObject rep) {
        JSONObject h = rep.optJSONObject("hetero");
        if (h == null) return null;
        final boolean ran = h.optBoolean("ran");
        if (!ran && h.optString("error").isEmpty()) return null;
        LinearLayout card = ui.card();
        LinearLayout head = ui.horizontal();
        LinearLayout titles = ui.vertical();
        titles.addView(ui.title("CPU + GPU split", 17));
        titles.addView(ui.caption("One SGEMM, rows shared by throughput · " + pretty(h.optString("size"))));
        head.addView(titles, Ui.weight(1f));
        if (ran) {
            boolean ok = h.optBoolean("verified");
            head.addView(ui.chip(ok ? "✓ verified" : "✕ mismatch", ok ? ui.success : ui.error,
                    ok ? ui.successContainer : ui.errorContainer));
        }
        card.addView(head);
        if (!ran) {
            TextView t = ui.body("Not run: " + h.optString("error"));
            t.setTextColor(ui.onSurfaceVariant);
            t.setPadding(0, ui.dp(8), 0, 0);
            card.addView(t);
            return card;
        }
        JSONObject w = h.optJSONObject("wall_ms");
        double split = w == null ? Double.NaN : w.optDouble("median", Double.NaN);
        double cpu = h.optDouble("cpu_alone_ms", Double.NaN), gpu = h.optDouble("gpu_alone_ms", Double.NaN);
        double speedup = h.optDouble("speedup_vs_best_single", Double.NaN);
        String gpuId = h.optString("gpu");
        String gpuDev = h.optString("gpu_device");

        TextView big = ui.title(String.format(Locale.getDefault(), "%.2f×", speedup), 28);
        big.setTextColor(speedup >= 1 ? ui.success : ui.onSurface);
        big.setPadding(0, ui.dp(8), 0, 0);
        card.addView(big);
        card.addView(ui.caption(speedup >= 1 ? "faster than the best single device" : "of the best single device's speed (overhead dominates here)"));

        double maxT = Math.max(split, Math.max(cpu, gpu));
        card.addView(timeBar("cpu alone", cpu, maxT, ui.familyColor("cpu")), ui.matchWrap(12, 0));
        card.addView(timeBar(gpuId + " alone", gpu, maxT, ui.familyColor(gpuId)), ui.matchWrap(8, 0));
        card.addView(timeBar("together", split, maxT, ui.primary), ui.matchWrap(8, 0));
        TextView rows = ui.caption("cpu " + h.optInt("cpu_rows") + " rows + " + gpuId + " " + h.optInt("gpu_rows") + " rows"
                + (gpuDev.isEmpty() ? "" : " (" + gpuDev + ")") + " · shorter is better");
        rows.setPadding(0, ui.dp(8), 0, 0);
        card.addView(rows);
        return card;
    }

    private View timeBar(String label, double t, double max, int color) {
        LinearLayout l = ui.vertical();
        LinearLayout top = ui.horizontal();
        top.addView(ui.text(label, 13, ui.onSurface), Ui.weight(1f));
        top.addView(ui.text(ms(t), 13, ui.onSurfaceVariant));
        l.addView(top);
        l.addView(bar(max > 0 ? t / max : 0, color), ui.matchWrap(4, 0));
        return l;
    }

    // ---- devices ------------------------------------------------------------------------------

    /** Rows for every backend, available first. Returns the number available. */
    int deviceRows(LinearLayout parent, JSONObject devices) {
        JSONArray bs = devices == null ? null : devices.optJSONArray("backends");
        if (bs == null) return 0;
        List<JSONObject> list = new ArrayList<>();
        for (int i = 0; i < bs.length(); ++i) if (bs.optJSONObject(i) != null) list.add(bs.optJSONObject(i));
        list.sort((a, b) -> Boolean.compare(b.optBoolean("available"), a.optBoolean("available")));
        int available = 0;
        for (JSONObject b : list) {
            if (b.optBoolean("available")) ++available;
            parent.addView(deviceRow(b), ui.matchWrap(2, 2));
        }
        return available;
    }

    private View deviceRow(JSONObject b) {
        final boolean avail = b.optBoolean("available");
        final String id = b.optString("id");
        LinearLayout row = ui.vertical();
        row.setMinimumHeight(ui.dp(56));
        row.setPadding(ui.dp(8), ui.dp(8), ui.dp(8), ui.dp(8));
        row.setBackground(ui.ripple(ui.rounded(0x00000000, 12), 12));

        LinearLayout line = ui.horizontal();
        View dot = new View(ui.ctx);
        dot.setBackground(ui.rounded(avail ? ui.familyColor(id) : ui.outline, 999));
        LinearLayout.LayoutParams dlp = new LinearLayout.LayoutParams(ui.dp(10), ui.dp(10));
        dlp.rightMargin = ui.dp(12);
        line.addView(dot, dlp);
        LinearLayout texts = ui.vertical();
        LinearLayout idLine = ui.horizontal();
        idLine.addView(ui.title(id, 14.5f));
        if (!b.optBoolean("genuine", true)) addChip(idLine, "stand-in", ui.onSurfaceVariant, ui.surfaceVariant);
        String cls = b.optString("device_class");
        if (avail && ("gpu".equals(cls) || "npu".equals(cls))) addChip(idLine, cls.toUpperCase(Locale.US), ui.onPrimaryContainer, ui.primaryContainer);
        texts.addView(idLine);
        String dev = b.optString("device");
        String api = b.optString("api");
        TextView sub = ui.caption(dev.isEmpty() ? api : dev + " · " + api);
        sub.setMaxLines(2);
        sub.setEllipsize(TextUtils.TruncateAt.END);
        texts.addView(sub);
        if (!avail) {
            TextView why = ui.text(b.optString("note"), 12.5f, ui.onSurfaceVariant);
            why.setMaxLines(3);
            why.setEllipsize(TextUtils.TruncateAt.END);
            texts.addView(why);
        }
        line.addView(texts, Ui.weight(1f));
        TextView state = ui.text(avail ? "Ready" : "Unavailable", 12, avail ? ui.success : ui.onSurfaceVariant);
        state.setPadding(ui.dp(8), 0, 0, 0);
        line.addView(state);
        row.addView(line);

        StringBuilder sb = new StringBuilder();
        appendKv(sb, "vendor", b.optString("vendor"));
        appendKv(sb, "driver", b.optString("driver"));
        appendKv(sb, "class", cls);
        appendKv(sb, "note", b.optString("note"));
        JSONObject det = b.optJSONObject("detail");
        if (det != null) {
            for (Iterator<String> it = det.keys(); it.hasNext(); ) {
                String k = it.next();
                appendKv(sb, k, det.optString(k));
            }
        }
        final TextView details = ui.monoBlock(sb.toString().trim());
        details.setVisibility(View.GONE);
        row.addView(details, ui.matchWrap(8, 0));
        row.setOnClickListener(x -> details.setVisibility(details.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE));
        row.setContentDescription(id + ", " + (avail ? "ready" : "unavailable") + ", " + dev + ". Tap for details.");
        return row;
    }

    private static void appendKv(StringBuilder sb, String k, String v) {
        if (v == null || v.isEmpty()) return;
        sb.append(k).append(": ").append(v).append('\n');
    }
}
