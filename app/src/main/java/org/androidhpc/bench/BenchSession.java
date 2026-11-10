// SPDX-License-Identifier: Apache-2.0
package org.androidhpc.bench;

/**
 * Process-wide state of the benchmark, kept outside the Activity so that a recreated Activity
 * (dark-mode switch, window resize, return from the background) shows the run in progress or its
 * results instead of losing them or starting a second native run. Main-thread access only, except
 * {@link #running}, which worker threads clear.
 */
final class BenchSession {
    private BenchSession() {}

    static volatile boolean running;
    static volatile boolean discovering;
    static long runStartedMs;

    /** Last discovery (backends) and last run, as returned by the native library. */
    static String devicesJson, devicesText;
    static String reportJson, reportText;
    static long reportElapsedMs;
    static String reportSavedPath;
}
