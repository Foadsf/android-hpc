// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;

/**
 * Small styled-view factory for the programmatic UI (no AndroidX, no XML layouts). Colours come
 * from res/values(-night)/colors.xml, so light and dark themes follow the system setting.
 * Every interactive view is at least 48 dp tall (Material touch-target minimum).
 */
final class Ui {
    static final int TOUCH_MIN_DP = 48;

    final Context ctx;
    final float density;
    final int bg, surface, surfaceVariant, outline, onSurface, onSurfaceVariant, primary, onPrimary,
            primaryContainer, onPrimaryContainer, success, successContainer, error, errorContainer,
            warning, warningContainer, track;
    final Typeface medium = Typeface.create("sans-serif-medium", Typeface.NORMAL);
    final Typeface mono = Typeface.MONOSPACE;

    Ui(Context c) {
        ctx = c;
        density = c.getResources().getDisplayMetrics().density;
        bg = color(R.color.bg);
        surface = color(R.color.surface);
        surfaceVariant = color(R.color.surface_variant);
        outline = color(R.color.outline);
        onSurface = color(R.color.on_surface);
        onSurfaceVariant = color(R.color.on_surface_variant);
        primary = color(R.color.primary);
        onPrimary = color(R.color.on_primary);
        primaryContainer = color(R.color.primary_container);
        onPrimaryContainer = color(R.color.on_primary_container);
        success = color(R.color.success);
        successContainer = color(R.color.success_container);
        error = color(R.color.error);
        errorContainer = color(R.color.error_container);
        warning = color(R.color.warning);
        warningContainer = color(R.color.warning_container);
        track = color(R.color.track);
    }

    int color(int id) { return ctx.getColor(id); }

    int dp(float v) { return Math.round(v * density); }

    /** Colour of a backend family ("vulkan:1" -> vulkan). */
    int familyColor(String backendId) {
        String f = backendId == null ? "" : backendId;
        int colon = f.indexOf(':');
        if (colon >= 0) f = f.substring(0, colon);
        switch (f) {
            case "cpu": return color(R.color.fam_cpu);
            case "vulkan": return color(R.color.fam_vulkan);
            case "opencl": return color(R.color.fam_opencl);
            case "gles": return color(R.color.fam_gles);
            case "nnapi": return color(R.color.fam_nnapi);
            default: return color(R.color.fam_other);
        }
    }

    // ---- drawables ----------------------------------------------------------------------------

    GradientDrawable rounded(int fill, float radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fill);
        d.setCornerRadius(dp(radiusDp));
        return d;
    }

    GradientDrawable outlined(int fill, int stroke, float radiusDp) {
        GradientDrawable d = rounded(fill, radiusDp);
        d.setStroke(Math.max(1, dp(1)), stroke);
        return d;
    }

    /** Ripple feedback over {@code content}, clipped to a rounded rect. */
    Drawable ripple(Drawable content, float radiusDp) {
        int rippleColor = (onSurface & 0x00FFFFFF) | 0x22000000;
        return new RippleDrawable(ColorStateList.valueOf(rippleColor), content, rounded(0xFF000000, radiusDp));
    }

    // ---- text ---------------------------------------------------------------------------------

    TextView text(CharSequence s, float sp, int color) {
        TextView t = new TextView(ctx);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        t.setLineSpacing(0, 1.12f);
        return t;
    }

    TextView title(CharSequence s, float sp) {
        TextView t = text(s, sp, onSurface);
        t.setTypeface(medium);
        return t;
    }

    TextView body(CharSequence s) { return text(s, 14, onSurface); }

    TextView caption(CharSequence s) { return text(s, 12.5f, onSurfaceVariant); }

    TextView sectionLabel(CharSequence s) {
        TextView t = text(s, 13, onSurfaceVariant);
        t.setTypeface(medium);
        t.setLetterSpacing(0.04f);
        t.setPadding(dp(4), dp(18), dp(4), dp(8));
        return t;
    }

    /** Small rounded label, e.g. "verified". Uses words, never colour alone. */
    TextView chip(CharSequence s, int fg, int bgColor) {
        TextView t = text(s, 12, fg);
        t.setTypeface(medium);
        t.setGravity(Gravity.CENTER);
        t.setPadding(dp(8), dp(3), dp(8), dp(3));
        t.setBackground(rounded(bgColor, 999));
        t.setSingleLine(true);
        return t;
    }

    TextView monoBlock(CharSequence s) {
        TextView t = text(s, 11.5f, onSurface);
        t.setTypeface(mono);
        t.setTextIsSelectable(true);
        t.setLineSpacing(0, 1.05f);
        t.setPadding(dp(12), dp(10), dp(12), dp(10));
        t.setBackground(rounded(surfaceVariant, 10));
        return t;
    }

    // ---- containers ---------------------------------------------------------------------------

    LinearLayout vertical() {
        LinearLayout l = new LinearLayout(ctx);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    LinearLayout horizontal() {
        LinearLayout l = new LinearLayout(ctx);
        l.setOrientation(LinearLayout.HORIZONTAL);
        l.setGravity(Gravity.CENTER_VERTICAL);
        return l;
    }

    /** Elevated-looking surface card (flat fill + hairline outline; cheap to draw). */
    LinearLayout card() {
        LinearLayout c = vertical();
        c.setBackground(outlined(surface, outline, 16));
        c.setPadding(dp(16), dp(14), dp(16), dp(14));
        return c;
    }

    LinearLayout tintedCard(int fill) {
        LinearLayout c = vertical();
        c.setBackground(rounded(fill, 16));
        c.setPadding(dp(16), dp(14), dp(16), dp(14));
        return c;
    }

    static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    LinearLayout.LayoutParams matchWrap(int topDp, int bottomDp) {
        LinearLayout.LayoutParams lp = matchWrap();
        lp.topMargin = dp(topDp);
        lp.bottomMargin = dp(bottomDp);
        return lp;
    }

    static LinearLayout.LayoutParams weight(float w) {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, w);
    }

    View divider() {
        View v = new View(ctx);
        v.setBackground(new ColorDrawable(outline));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, dp(1)));
        lp.topMargin = dp(4);
        lp.bottomMargin = dp(4);
        v.setLayoutParams(lp);
        return v;
    }

    // ---- buttons ------------------------------------------------------------------------------

    private Button baseButton(CharSequence label) {
        Button b = new Button(ctx);
        b.setText(label);
        b.setAllCaps(false);
        b.setTypeface(medium);
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        b.setMinHeight(dp(TOUCH_MIN_DP));
        b.setMinimumHeight(dp(TOUCH_MIN_DP));
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setStateListAnimator(null);
        b.setElevation(0);
        b.setPadding(dp(16), dp(10), dp(16), dp(10));
        return b;
    }

    Button filledButton(CharSequence label) {
        Button b = baseButton(label);
        styleFilled(b, primary, onPrimary);
        return b;
    }

    void styleFilled(Button b, int fill, int fg) {
        b.setTextColor(fg);
        b.setBackground(ripple(rounded(fill, 14), 14));
    }

    Button tonalButton(CharSequence label) {
        Button b = baseButton(label);
        b.setTextColor(onPrimaryContainer);
        b.setBackground(ripple(rounded(primaryContainer, 12), 12));
        return b;
    }

    Button outlinedButton(CharSequence label) {
        Button b = baseButton(label);
        b.setTextColor(primary);
        b.setBackground(ripple(outlined(0x00000000, outline, 12), 12));
        return b;
    }

    Button textButton(CharSequence label, int fg) {
        Button b = baseButton(label);
        b.setTextColor(fg);
        b.setBackground(ripple(rounded(0x00000000, 12), 12));
        return b;
    }

    /** Dims a disabled button's label without changing layout. */
    static void setEnabled(View v, boolean enabled) {
        v.setEnabled(enabled);
        v.setAlpha(enabled ? 1f : 0.45f);
    }

    /** Row of equal-width buttons with gaps. */
    LinearLayout buttonRow(View... buttons) {
        LinearLayout row = horizontal();
        for (int i = 0; i < buttons.length; ++i) {
            LinearLayout.LayoutParams lp = weight(1f);
            if (i > 0) lp.leftMargin = dp(8);
            row.addView(buttons[i], lp);
        }
        return row;
    }

    // ---- switches -----------------------------------------------------------------------------

    /**
     * A full-width Switch: title and description on the left, the toggle on the right. One native
     * control, so the whole row is the touch target and TalkBack reads "title, description, switch, on".
     */
    Switch switchRow(LinearLayout parent, CharSequence title, CharSequence description, boolean checked) {
        SpannableStringBuilder sb = new SpannableStringBuilder(title);
        if (description != null && description.length() > 0) {
            int start = sb.length() + 1;
            sb.append('\n').append(description);
            sb.setSpan(new RelativeSizeSpan(0.86f), start, sb.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            sb.setSpan(new ForegroundColorSpan(onSurfaceVariant), start, sb.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        Switch sw = new Switch(ctx);
        sw.setText(sb);
        sw.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        sw.setTextColor(onSurface);
        sw.setLineSpacing(0, 1.12f);
        sw.setChecked(checked);
        sw.setMinHeight(dp(56));
        sw.setMinimumHeight(dp(56));
        sw.setSwitchPadding(dp(16));
        sw.setPadding(dp(4), dp(8), dp(4), dp(8));
        sw.setBackground(ripple(rounded(0x00000000, 12), 12));
        parent.addView(sw, matchWrap());
        return sw;
    }
}
