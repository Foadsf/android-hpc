# android-hpc

An open-source proof of concept for **heterogeneous high-performance computing on Android**: one
benchmark suite, five kernels, run on the CPU, Vulkan compute (SPIR-V), OpenCL, OpenGL ES 3.1
compute and NNAPI, with every result checked against an fp64 CPU reference. It ships as an APK and
as a command-line tool (host PC or `adb shell`), plus a SysML v2 model of the requirements and
architecture.

It started from a Grok-generated SysML v2 model and a long Gemini audit of it. Both were
fact-checked and then re-implemented from scratch; see [docs/FACT_CHECK.md](docs/FACT_CHECK.md).

> **Read this first.** This is a measurement and teaching tool, not a tuned library. The kernels
> are straightforward tiled implementations and do **not** reach hardware peak. The project's claim
> is that what it reports is *real and verified*, not that it is fast. The table under
> "Verification status" says which backend has run on which hardware.

## What is in it

| Backend | How it works | Kernels | Notes |
|---|---|---|---|
| `cpu` | C++17 persistent thread pool, dynamic scheduling, packed-panel register-tiled GEMM, auto-vectorised | all 5 | pins workers only on heterogeneous (big.LITTLE) CPUs |
| `vulkan:N` | one backend per Vulkan device; SPIR-V built from GLSL at build time with `glslc`; GPU timestamp queries | all 5 | unified-memory GPUs use host-visible memory directly, discrete GPUs use staging; optional experimental `AHardwareBuffer` import |
| `opencl:N` | runtime `dlopen` of the vendor ICD (OpenCL is not an NDK API); event profiling | all 5 | needs `<uses-native-library>` on Android 12+, declared in the manifest |
| `gles` | OpenGL ES 3.1 compute shaders over a headless EGL pbuffer | all 5 | Android only; uses `GL_EXT_disjoint_timer_query` when present |
| `nnapi:N` | one NNAPI `FULLY_CONNECTED` op per NNAPI device, via `dlopen` | `dense` | NNAPI is deprecated since Android 15; a device called `nnapi-reference` is a CPU, not an NPU, and is labelled so |
| `sycl-shim` | SYCL-shaped API on the CPU thread pool | all 5 | **Not SYCL.** Marked `genuine=false` in every report. Real SYCL is not packaged for Android |

Kernels: `sgemm` (compute-bound), `dense` (GEMM + bias + ReLU, one NN layer), `conv3x3`, `reduce`
(two-pass multi-workgroup sum), `triad` (STREAM, bandwidth-bound).

Also: a **CPU + GPU split** experiment (rows of one SGEMM run concurrently on both, output verified),
CPU topology/ISA probing, per-device capability probes (memory model, subgroup size, fp64 support,
workgroup limits), and JSON + text reports with min/median/mean/p95/stddev for wall time and, where
the API provides it, on-device time.

## Get the APK

A debug-signed APK is built by CI on every push (Actions artifact `android-hpc-debug-apk`) and the
build below produces it locally at `app/build/outputs/apk/debug/app-debug.apk`.

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or copy the file to the phone and open it (allow "install unknown apps" for your file manager).
Open **Android HPC**, pick a problem size, press **Run benchmark**. Results appear as one ranked bar
chart per kernel (tap a row for wall/device timings, prepare time and error); the devices list shows
every backend and why an unavailable one is unavailable. Every finished run is saved automatically to
`/sdcard/Android/data/org.androidhpc.bench/files/reports/hpc-<timestamp>.json` (fetch with
`adb pull`); **Export JSON** saves a copy anywhere through the system file picker, **Share** and
**Copy** send the text report. The UI follows the system light/dark theme, survives rotation and
window resizing mid-run, and asks before Back abandons a running benchmark.

arm64-v8a is the real target; x86_64 is included for emulators.

## Build

Requirements: JDK 17, Android SDK platform 35 + build-tools 35, **NDK 26.1.10909125**, CMake 3.22.1.
Gradle is provided by the wrapper.

```bash
./gradlew assembleDebug              # APK (Windows: gradlew.bat)
```

Command-line tool for Android (`adb shell`), PowerShell:

```powershell
.\scripts\build-android-cli.ps1 -Abi arm64-v8a
adb push dist\hpc_cli-arm64-v8a /data/local/tmp/hpc_cli
adb shell chmod +x /data/local/tmp/hpc_cli
adb shell /data/local/tmp/hpc_cli --preset small --json /data/local/tmp/out.json
```

Host build (Windows/Linux). It runs the CPU backend plus whatever Vulkan and OpenCL devices the
machine has, which is the fastest way to test the code. Needs a C++17 compiler, CMake, the Vulkan
headers and `glslc` (the Vulkan SDK includes both):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build/core/hpc_cli --preset medium --json report.json     # .../Release/hpc_cli.exe with MSVC
```

`hpc_cli --help` lists options. Exit status 0 means every attempted run matched the fp64 reference.

## How to read a report

* `wall ms`: host-measured time of one blocking run, including launch and sync.
* `dev ms`: device-side time (Vulkan timestamps, OpenCL events, GL timer query, NNAPI on-hardware
  duration) when the API reports it *and* every sample is consistent with the wall time of the same
  run; otherwise `-` and the wall time is used.
* `GFLOP/s`, `GB/s`: use device time if available, otherwise wall time. `GB/s` counts compulsory
  traffic only, so it understates real traffic for the tiled kernels; treat it as meaningful for
  `triad` and `reduce` only.
* `ok`: the result matched the fp64 oracle within a per-kernel tolerance. A failed or mismatching run
  makes the CLI exit non-zero.
* Pipeline compilation and uploads happen in `prepare()` and are never inside a timed region.
* Energy is **not** measured.

A real example (Windows host, Intel UHD + NVIDIA RTX A2000, `--preset medium`) is in
[docs/example-report-windows-host.json](docs/example-report-windows-host.json).
Excerpt (sgemm 512³): CPU 27.8, Intel iGPU via Vulkan 67, via OpenCL 68, NVIDIA via Vulkan 481,
via OpenCL 462 GFLOP/s; NVIDIA triad ~163 GB/s. Phones will look very different.

## Verification status

| Component | Status | How |
|---|---|---|
| CPU, SYCL-shim | **Run and verified** | `hpc_tests` on x86-64 (MSVC); also on an arm64 phone (8 cores, NEON), 0.1.1 log of 2026-11-08, all results matched the fp64 reference |
| Vulkan and OpenCL on a phone GPU | **Run and verified on two devices** | Arm Mali-G71 (Vulkan 1.0.97, OpenCL 2.0) and Arm Mali-G76 (Vulkan 1.1.177, OpenCL 3.0), 0.1.1 logs of 2026-11-08: sgemm, conv3x3, reduce, triad and dense matched the reference on both (GPUs as the drivers report them; the phone models are not in the partial logs) |
| Vulkan (unified-memory path) | **Run and verified** | Intel UHD (integrated) on Windows |
| Vulkan (discrete/staged path) | **Run and verified** | NVIDIA RTX A2000 on Windows |
| Vulkan (third driver stack) | **Run and verified in CI** | Mesa llvmpipe (software Vulkan 1.4) on a GitHub Linux runner |
| OpenCL | **Run and verified** | Intel and NVIDIA ICDs on Windows; PoCL (CPU) in Linux CI |
| SPIR-V | **Validated** | `spirv-val --target-env vulkan1.0`, one ctest per shader |
| Verifier | **Negative-controlled** | corrupt element, NaN and wrong length are rejected |
| Odd sizes / bounds | **Verified** | 7, 33, 61, 77, 1001, 100003 and a non-square 65x77x77 slice on every available backend |
| APK build | **Built, signature, 16 KB alignment, manifest checked** | `apksigner`, `zipalign -P 16`, `aapt2 dump badging` |
| Android runtime (JNI, UI) | **0.1.0 crashed on launch on the first real phone; 0.1.1 ran two full 22-result benchmark runs without a crash, all verified** | the 0.1.0 crash cause was never captured (no log existed), so which of the 0.1.1 fixes cured it is unknown. The 0.1.1 log was shared in part (the start of the first run is missing) |
| OpenGL ES backend | **Results verified on the Mali-G76 (Samsung SM-N970F) in 0.1.1 and 0.1.2 (28 results, all verified; the CPU+GLES split ran); on the Mali-G71 (Samsung SM-N950F, 128 invocations) 0.1.3 ran GLES with 8x8 / 128-wide workgroups and every result matched the reference** | the Mali-G71 reports 128 invocations per workgroup, so 0.1.1 marked GLES unavailable there; since 0.1.2 the shaders take their workgroup shape from the driver limit (8x8 / 128 on the G71, 16x16 / 256 on the G76). Both Mali GL timer queries return nonsense for compute (about 200 ns-2 us for multi-ms dispatches). Up to 0.1.3 a fixed 1 us cut-off let some of those through to the "dev ms" column; since 0.2.0 every device-timer sample of every API is checked against the wall time of the same run (it must be neither longer nor 50x shorter) and one bad sample discards the timer for that result (`device_samples_rejected` in the JSON). 0.2.0 has not yet been run on a phone |
| NNAPI backend | **Run and verified on two devices** | Mali-G71 phone: only `nnapi-reference` (CPU). Mali-G76 phone: `eden-drv` (an accelerator) plus `nnapi-reference`; dense matched the reference on all of them |
| Vulkan `AHardwareBuffer` import | **Succeeded on the Mali-G76 (Vulkan 1.1); fell back to copy on the Mali-G71 (Vulkan 1.0.97)** | the G71 log gives the reason: `VK_ANDROID_external_memory_android_hardware_buffer not enabled` (that driver exposes 21 device extensions; the G76 exposes 66). Experimental |
| SysML model | **Parsed and checked with the FLOSS `sysml-v2-lsp`: 0 errors, 1 warning** | `node scripts/check-sysml.cjs model/AndroidHPC.sysml` (also in CI); it found and I fixed two real syntax defects. Not checked against the OMG reference implementation |
| CI | runs on GitHub Actions | Linux: software Vulkan (lavapipe) + software OpenCL (PoCL); APK build |

If you run it on a phone, please open an issue with the saved JSON. A single report from a device
turns a "not yet run" row into evidence.

## Layout

```
core/            portable C++17 library (backends, harness, reference oracle), hpc_cli, tests
core/shaders/    GLSL kernel bodies shared by Vulkan (SPIR-V) and GLES (ES 3.1 text)
jni/, app/       JNI bridge and the single-activity Java app (no AndroidX)
model/           SysML v2 model: requirements, architecture, allocation, verification
docs/            FACT_CHECK.md, ARCHITECTURE.md, an example report
scripts/         Android CLI build script
```

Adding a backend: implement `hpc::Backend` and `hpc::Execution` (`core/include/hpc/backend.hpp`),
add a `discoverX()` call in `core/src/discover.cpp`. Adding a kernel: a `Problem` case, an oracle in
`reference.cpp`, and a kernel per backend.

## Debugging a crash

The app writes synchronous (`fsync`ed) debug logs and durable vendor-driver breadcrumbs to app-specific external storage so that even immediate native crashes (`SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGFPE`, `SIGABRT`) or vendor driver hangs leave a diagnostic trail on disk.

### Where logs are stored on the phone

- **Logs directory:** `/sdcard/Android/data/org.androidhpc.bench/files/logs/`
  (falls back to internal `<getFilesDir()>/logs/` if external storage is unavailable)
- **Per-session log file:** `hpc-<yyyyMMdd-HHmmss>.log` (the newest 10 logs are kept; the absolute path is shown under **Diagnostics and logs** in the app and printed as the first line of the log)
- **Breadcrumb file:** `breadcrumb.txt` (holds the last step entered before a crash, e.g. `discover:opencl:begin`)
- **Session marker:** `session.open` (created at startup and deleted on clean `onDestroy` when `isFinishing()`)

### Exact `adb` commands to collect crash evidence

Pull all persistent logs and breadcrumb files from the phone:
```sh
adb pull /sdcard/Android/data/org.androidhpc.bench/files/logs/
```

Dump the Android native crash buffer (tombstone summary). The app's own signal handler writes its
report and then hands the signal to the handler that was installed before it (debuggerd), so the
system tombstone is still produced:
```sh
adb logcat -d -b crash
```

Stream or dump the application's logcat tag (`android-hpc`):
```sh
adb logcat -s android-hpc
```

### How automatic crash quarantine works

1. **Breadcrumbs around vendor driver calls:** Immediately before and after every risky vendor-driver step, native code writes and `fsync`s a breadcrumb using the exact forms:
   - `discover:<family>:begin` / `discover:<family>:end`
   - `prepare:<backendId>:<kernel>:begin` / `prepare:<backendId>:<kernel>:end`
   - `run:<backendId>:<kernel>:begin` / `run:<backendId>:<kernel>:end`
2. **Unclean exit detection:** At startup, `DebugLog` checks whether `session.open` is still present from the previous launch.
3. **Quarantine on `:begin`:** If `session.open` exists, the app reads `breadcrumb.txt` and the last 40 lines of the previous log. If the last breadcrumb ends in `:begin`, the corresponding backend family (`vulkan`, `opencl`, `gles`, or `nnapi`) is persisted in `SharedPreferences` and passed to native code via `Native.setDisabledBackends(csv)` before hardware discovery begins.
4. **Recovery Banner & Re-enable:** On the next launch after a crash, the app shows a recovery card naming the driver step that never finished and the backend family that is now skipped; **Show log** expands the previous log path, the last breadcrumb and the last 40 lines of the previous log. Tapping **Re-enable** clears the quarantine and re-probes all backends.

## Known limitations

* CPU kernels are portable C++, not hand-tuned NEON/SVE; GPU kernels are classic shared-memory tiling
  (16x16 by default; GLES and OpenCL drop to 8x8 / 128 or 64 work-items when the driver limit for that
  kernel is lower, which OpenCL reports per kernel through `CL_KERNEL_WORK_GROUP_SIZE`).
* One GPU queue; no subgroup, fp16/int8 or cooperative-matrix paths yet.
* The CPU+GPU split only pairs the CPU with a real GPU or NPU (`device_class` in the report). A
  CPU-hosted runtime such as Mesa llvmpipe, SwiftShader or PoCL would only compete for the same cores.
* The CPU+GPU split uses a throughput-proportional static partition measured from earlier runs.
* No energy or thermal measurement. Sustained-load throttling is visible only as run-to-run drift.
* Vulkan targets SPIR-V 1.0 / Vulkan 1.0 for compatibility, which forgoes newer features.

## License

Apache-2.0, see [LICENSE](LICENSE). Contributions are accepted under the same license.
