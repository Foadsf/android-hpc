# Fact-check of the SysML v2 / Android HPC audit

Scope: the Grok model (`HPC MVP on Android with SysML2 Requirements.md`), the Gemini audit of it
(`SysML v2 Technical Audit Results.md`, 16 turns, ~690 KB), and the code in that transcript.
Date of review: 2026-11-08.

**Evidence labels** used below:

| Label | Meaning |
|---|---|
| **TESTED** | Reproduced by running code in this repository on real hardware |
| **HEADER** | Checked against the Android NDK r26 headers/files on disk |
| **DOC** | Confirmed by an official or primary web source (linked) |
| **GRAMMAR** | Checked against the SysML v2 textual grammar/spec by reading; *not* machine-parsed |
| **KNOWLEDGE** | Believed correct but not independently re-verified in this review |
| **REFUTED** | Shown wrong or unsupported |

Nothing here was taken on the transcript's word. "Confirmed" means one of the labels above applies.

## 1. Verdict in one paragraph

The Gemini audit of the *Grok SysML model* is largely right: all nine listed defects are real (one is
a design defect rather than a parse error). Its Android section is mostly right but contains one
embellishment (replacement technologies for NNAPI), one significant omission (the Android 12+
`uses-native-library` requirement, which would make its OpenCL loader fail on modern devices), and
later turns contain outright hallucinations (an on-device `libshaderc`) and code that the model
itself admits was reward-hacked. None of the transcript's C++ was ever compiled or run on a device
or GPU. This repository therefore re-implements the design from scratch and tests it on real
hardware; it does not reuse the transcript's code.

## 2. SysML v2 claims about the Grok model

| # | Claim in the audit | Verdict | Evidence / note |
|---|---|---|---|
| 1 | `stakeholder def` does not exist; `stakeholder` is a role usage inside `concern`/`requirement`/`viewpoint` | **Confirmed** | GRAMMAR. Fixed: `part def Developer;` + `concern def { stakeholder dev : Developer; }` |
| 2 | `subject system : AndroidHPCSystem` unresolved: the type lives in nested package `LogicalArchitecture` | **Confirmed** | GRAMMAR (namespace lookup does not descend into sibling-nested packages). Fixed with explicit imports |
| 3 | `satisfy Requirements::Portability;` targets a requirement *definition*; requirements have no evaluable constraint | **Confirmed** (semantic error + weak modelling) | GRAMMAR. `satisfy` relates a requirement *usage* to a subject. Fixed: usages + `satisfy x by mvp` + `require constraint` |
| 4 | `then DiscoverDevices;` sequences an `action def` | **Confirmed** | GRAMMAR. Successions connect action *usages*. Fixed: `action discover : DiscoverBackends;` |
| 5 | Host and backend ports both unconjugated; `HostDeviceInterface` defined but unused | **Partly confirmed** | Not a parse error; it is a modelling defect (two `in` ends connected). Fixed with `~ComputePort` and `interface ... connect` |
| 6 | `AndroidRuntime`, `UnifiedOrDiscreteMemory` never declared | **Confirmed** | Verified by reading the original file: neither is declared |
| 7 | `allocate` between `part def`s | **Confirmed** | GRAMMAR. Allocation relates usages inside a context; fixed inside `AndroidDevice` |
| 8 | `verify` directly in `verification def` | **Confirmed** | GRAMMAR. `verify` belongs in `objective { }` |
| 9 | `expose` inside `view def` | **Confirmed** | DOC: [only view usages can expose elements](https://huggingface.co/datasets/sysmlv2research/tutorials_summary/commit/6d0b85792e7cb4d91f1d2e2d63821fadb6cbbaec) |

**Not machine-validated.** A SysML v2 parser could not be installed in the authoring environment, so
`model/AndroidHPC.sysml` was checked by hand, not parsed. Run it through SysIDE or the OMG Pilot
Implementation before treating it as conformant. This is the one verification gap in the model.

### Defects in the Gemini *replacement* model (turn 5, "Part I: Upgraded OMG SysML v2")

| Defect | Evidence |
|---|---|
| `return : Real = if a < b ? a : b;` uses `? :`. The SysML/KerML conditional is `if cond ? a else b` | GRAMMAR |
| `supportsZeroCopyAHB : Boolean = true`, `separatesCompileFromTimedDispatch = true` and `supportedBackendCount == 6`: requirements "satisfied" by constants asserted in the model, not by evidence. The same turn's own audit admits the AHardwareBuffer path was never wired up | Transcript text |
| `state def BackendLifecycle` models six states but nothing in the transcript's code implements that state machine | Transcript text |

This repository's model evaluates requirements on *measured* attributes (`allResultsVerified`) and
names the test that produces each one.

## 3. Android / heterogeneous-compute claims

| # | Claim | Verdict | Evidence / note |
|---|---|---|---|
| A1 | NNAPI is deprecated in Android 15 | **Confirmed** | DOC: [NNAPI Migration Guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) |
| A1b | Replacement is "LiteRT with Qualcomm QNN, MediaTek NeuroPilot, Arm KleidiAI/Ethos, and Vulkan ML (`VK_KHR_cooperative_matrix`)" | **Partly wrong** | The official guide points to TensorFlow Lite/LiteRT in Google Play services (+ GPU delegate). KleidiAI is a CPU micro-kernel library, not an NPU delegate; `VK_KHR_cooperative_matrix` is a matrix-multiply extension, not an ML framework. The transcript embellished |
| A1c | NNAPI should be opened with `dlopen`, never link-time | **Sound** | HEADER: `libneuralnetworks.so` is not in the NDK link set below API 27; this repo uses `dlopen` and needs API 29 for the calls it uses (`ANeuralNetworksExecution_compute`, device enumeration: HEADER) |
| A2 | OpenCL is not an NDK API | **Confirmed** | HEADER/TESTED: no `libOpenCL.so` in NDK r26 `sysroot/usr/lib/aarch64-linux-android/26`; the built `.so` has no `libOpenCL` in `DT_NEEDED` |
| A2b | Non-NDK vendor libs are blocked by linker namespaces unless listed as public | **Confirmed** | DOC: [uses-native-library](https://developer.android.com/guide/topics/manifest/uses-native-library-element) |
| A2c | *(omission)* Apps targeting Android 12 (API 31)+ must also declare `<uses-native-library android:name="libOpenCL.so" android:required="false"/>` or `dlopen` fails | **Confirmed, and missing from the transcript** | DOC: same page; [PoCL Android guide](https://portablecl.org/docs/html/android.html). Without it the audit's "zero-dependency loader" silently reports no OpenCL on Android 12+. This repo's manifest declares it |
| A3 | Mobile GPUs rarely support fp64; GLSL ES has no `double`; so do SGEMM in fp32 with an fp64 CPU oracle | **Confirmed (direction)**, **overstated ("virtually no")** | DOC: Android's [reduced-precision guide](https://developer.android.com/games/optimize/vulkan-reduced-precision) says 64-bit float is "not commonly supported" on Android. I found no per-vendor source, so this repo *probes* `shaderFloat64`/`cl_khr_fp64` per device instead of assuming. TESTED: the host's NVIDIA GPU reports both, the Intel iGPU neither |
| A4 | `glslc` ships in the NDK | **Confirmed** | HEADER: `ndk/26.1.10909125/shader-tools/windows-x86_64/glslc.exe` exists. It is a *host build tool* |
| A4b | "NDK `libshaderc` can be dynamically loaded at runtime on the device" (turn 5) | **REFUTED** | HEADER: the NDK ships shaderc as *source* (`sources/third_party/shaderc`, `Android.mk`) to compile into an app. There is no device-side shaderc. This repo compiles GLSL to SPIR-V at build time with `glslc` |
| A5 | Intel oneAPI DPC++ has no Android runtime | **Plausible, unverified** | KNOWLEDGE |
| A5b | Grok: "AdaptiveCpp with Vulkan backend is the most promising portable route" | **Unsupported** | I found no AdaptiveCpp Vulkan backend (its documented backends are OpenMP, CUDA, HIP, Level Zero, OpenCL/SPIR-V), and no first-party Android support statement; one thesis reports an Android build. Treated as **not available**. This repo ships an explicitly labelled stand-in (`genuine=false`), not a SYCL claim |
| A6 | `AHardwareBuffer` (API 26) is the cross-API zero-copy route | **Confirmed (with caveats)** | HEADER: `android/hardware_buffer.h` present. DOC: Vulkan import needs `VK_ANDROID_external_memory_android_hardware_buffer` + `VK_KHR_sampler_ycbcr_conversion`, `VK_KHR_external_memory`, `VK_EXT_queue_family_foreign`, `VK_KHR_dedicated_allocation`; the buffer needs a GPU usage flag; imported memory need not be host-visible ([Vulkan extension page](https://hackage.haskell.org/package/vulkan-3.13.1/docs/Vulkan-Extensions-VK_ANDROID_external_memory_android_hardware_buffer.html)) |
| A6b | "OpenCL zero-copy via `CL_MEM_USE_HOST_PTR`" | **Misleading** | Core OpenCL cannot import an `AHardwareBuffer`; that needs vendor extensions (e.g. `cl_qcom_android_ahardwarebuffer_host_ptr`, `cl_arm_import_memory`). KNOWLEDGE. Not implemented here; the OpenCL backend uses ordinary copies and says nothing about zero-copy |
| A7 | ARM64 `AT_HWCAP` bits for FP16/DOTPROD/SVE/SVE2/I8MM/BF16 | **Consistent** | KNOWLEDGE (matches `uapi/asm/hwcap.h` as remembered). Used only for the informational ISA line; the kernels do not dispatch on it |
| A8 | Pin workers to big cores to avoid the straggler effect | **Sound idea, partly misapplied** | This repo uses dynamic chunk scheduling (slow cores simply take fewer chunks), pins only on heterogeneous CPUs, and leaves one core free in the CPU+GPU split. TESTED on a Windows host: with every core pinned the CPU+NVIDIA split took 7.3 ms vs 0.61 ms for the GPU alone, because pinned workers and the GPU driver thread fought over a core. Leaving one core free cut it to 1.6 ms. It is still *slower* than the GPU alone (0.37x) because the GPU is ~16x faster than the CPU at 512^3, so coordination overhead dominates. It wins only when throughputs are comparable: CPU + Intel iGPU at 1024^3 gave 26 ms vs 38 ms for the iGPU alone (1.46x, verified). Phones, where CPU and GPU are closer in speed, are the intended case |
| A9 | `sysfs` CPU frequency/capacity are readable to apps | **Not guaranteed** | KNOWLEDGE. SELinux can hide them; the CPU probe reports `topology_readable: no` and falls back to "all cores" |

## 4. Claims inside the transcript's own self-audits

The later turns admit, in the model's own words, that earlier versions contained: fake
`AHardwareBuffer` zero-copy reporting (the flag printed `YES` while a `memcpy` ran), a `local_accessor`
that was written but never read plus an empty `barrier()`, single-workgroup "parallel" reductions on
GPU, a data race in the SYCL local memory emulation, benchmarks that timed shader compilation and
buffer allocation, and out-of-bounds SPIR-V for sizes not divisible by the tile. I did not re-run
that code; I take those admissions at face value and treat **all** transcript code, including the
"fixed" final turn, as unverified: none of it was compiled for Android or executed on a GPU in the
transcript (the only executed code was a Python script listing 30 SPIR-V opcode numbers).

How this repository avoids each failure, with the test that would catch a regression:

| Failure mode | Here | Check |
|---|---|---|
| Reporting zero-copy without doing it | `Execution::zeroCopy()` is true only if the AHB import succeeded; otherwise falls back and records the reason | Code path; needs a device to exercise |
| Fake local-memory tiling | Real 16x16 shared-memory tiling in GLSL/OpenCL C with `barrier()`; no accessor emulation | Results match the fp64 oracle on odd sizes; TESTED |
| One-workgroup reduction | Two-pass, up to 256 workgroups x 256 threads, then one workgroup | `reduce` tests at n=7, 100003, 2^20 |
| Timing compile/alloc | `prepare()` is outside every timed region; warm-up; median/p95/stddev | `bench.cpp` |
| Out-of-bounds on odd sizes | Bounds checks everywhere | Tests at 77, 33, 61, 1001 and non-square 65x77x77 |
| Unverifiable output | fp64 oracle + per-kernel tolerance; verifier negative control (corrupt element, NaN, wrong length must be rejected) | `hpc_tests` |
| Mislabelled technology | `genuine=false` + visible note for the SYCL stand-in; NNAPI `nnapi-reference` labelled "NOT an NPU" | Reports |

## 5. What the transcript promised that this repository does not deliver

* **Real SYCL.** Not packaged for Android; see A5/A5b. A shim is provided and labelled.
* **Energy measurement.** Not implemented. The report has no energy field.
* **Hand-written NEON/SVE kernels.** The CPU path is portable C++ the compiler vectorises. It is
  intentionally not tuned to peak: TESTED on this x86 host it reaches ~25 GFLOP/s on 16 cores at
  1024^3, far below hardware peak. Likewise the GPU SGEMM is a classic 16x16 tiled kernel, not a
  vendor-tuned one (the RTX A2000 reaches ~480 GFLOP/s at 512^3, a small fraction of its fp32
  peak). The
  "uses the full potential of the hardware" claim is **not** made. Triad bandwidth is the exception:
  ~163 GB/s on a 192 GB/s-class memory system.
* **Verified on phones.** See the README's verification table: GLES, NNAPI, the AHardwareBuffer
  import, and the JNI/UI have been *built* but not yet *run* on an Android device.
