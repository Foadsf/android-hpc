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
        } catch (UnsatisfiedLinkError e) {
            loadError = String.valueOf(e.getMessage());
        }
    }

    /** Probe every backend; returns the human-readable report. */
    static native String discover(boolean bigOnly);

    /** Run the suite; returns the JSON report. */
    static native String run(String preset, int warmup, int iters, String backendsCsv, String kernelsCsv,
                             boolean zeroCopy, boolean hetero, boolean bigOnly);

    static native String progress();
    static native String lastText();
    static native String lastJson();
    static native void cancel();
    static native String version();
}
