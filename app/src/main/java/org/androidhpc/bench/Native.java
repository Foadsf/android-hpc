// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

/** JNI surface of libhpcjni.so. All methods are thread-safe; run() blocks until finished. */
final class Native {
    private Native() {}

    static boolean loaded;
    static String loadError = "";

    static {
        try {
            System.loadLibrary("hpcjni");
            loaded = true;
        } catch (Throwable e) {
            loadError = String.valueOf(e);
        }
    }

    /** Initialize native file + breadcrumb logging and async-signal-safe crash handlers. */
    static native void initLogging(String logPath, String breadcrumbPath, boolean verbose);

    /** Toggle verbose native logging at runtime. */
    static native void setVerbose(boolean verbose);

    /** Read the last recorded breadcrumb from native memory. */
    static native String lastBreadcrumb();

    /** Set comma-separated backend families (vulkan, opencl, gles, nnapi) to skip. */
    static native void setDisabledBackends(String csv);

    /** Probe every backend; returns the JSON report (backends only). {@link #lastText()} has the text form. */
    static native String discover(boolean bigOnly);

    /** Run the suite; returns the JSON report, or {"error":...}. Only one discover/run executes at a time. */
    static native String run(String preset, int warmup, int iters, String backendsCsv, String kernelsCsv,
                             boolean zeroCopy, boolean hetero, boolean bigOnly);

    static native String progress();
    static native String lastText();
    static native String lastJson();
    static native void cancel();
    static native String version();
}
