# Architecture

The authoritative architecture is the SysML v2 model, [`model/AndroidHPC.sysml`](../model/AndroidHPC.sysml).
This page maps it to code.

```
 JNI / CLI / tests
        |
   runBenchmarks (bench.cpp)      problem -> fp64 oracle -> for each backend:
        |                          prepare (untimed) -> warm-up -> timed runs -> read -> verify
        v
   Backend / Execution  (backend.hpp)
   +-- cpu ---- ThreadPool (cpu.cpp, cpu_pool) + cpu_backend.cpp
   +-- sycl-shim ---- sycl_shim.hpp on the same pool
   +-- vulkan:N --- vulkan_backend.cpp   dlopen loader, SPIR-V (build time), timestamps
   +-- opencl:N --- opencl_backend.cpp   dlopen ICD, minimal ABI, event profiling
   +-- gles ------- gles_backend.cpp     headless EGL, ES 3.1 compute, timer query
   +-- nnapi:N ---- nnapi_backend.cpp    dlopen, FULLY_CONNECTED per device
```

| SysML element | Code |
|---|---|
| `ComputeBackend`, `ComputePort` | `hpc::Backend`, `hpc::Execution` |
| `RunBenchmarkSuite` actions | `discoverBackends`, `Backend::prepare`, timed `Execution::run`, `verify`, `toJson/toText` |
| `MemoryManager` | per-backend buffer handling; Vulkan picks unified vs staged by device type/heaps |
| `genuine` attribute | `BackendInfo::genuine` |
| `allResultsVerified` | `Report::allVerified` (CLI exit status) |
| `R-SPIRV` | `core/CMakeLists.txt` glslc step + `spirv_validates_*` ctests |
| `R-HETERO` | the split experiment at the end of the `sgemm` loop in `bench.cpp` |

## Design rules that are enforced, not just written down

1. **Nothing vendor-specific is linked.** Vulkan, OpenCL and NNAPI are opened with `dlopen`; the
   native library's `DT_NEEDED` is libc/libm/libdl/liblog/libEGL/libGLESv3/libandroid only.
2. **Compile and upload are outside the timed region.** `prepare()` does them; runs are re-submitted
   (Vulkan records its command buffer once).
3. **Every result is verified** against a double-precision CPU oracle, and the verifier is itself
   tested with a negative control.
4. **Labels are truthful.** Stand-ins carry `genuine=false`; unavailable backends keep their reason;
   zero-copy is only reported if the import call succeeded.
5. **GPU work is not under-dispatched.** Reductions use up to 256 workgroups x 256 threads; matrix
   kernels cover the whole grid with bounds checks.

## Memory model

* Unified memory (phones, integrated GPUs): buffers live in `DEVICE_LOCAL | HOST_VISIBLE` memory and
  are written directly; no staging copy.
* Discrete GPUs: `DEVICE_LOCAL` buffers plus a host-visible staging buffer; uploads and read-back use
  a copy command.
* Experimental (`--zero-copy`, Android, Vulkan): inputs are allocated as `AHardwareBuffer` BLOBs,
  filled by the CPU and imported into Vulkan with `VK_ANDROID_external_memory_android_hardware_buffer`.
  On any failure it falls back to the normal path and records the reason in the backend details.

## Why GLSL -> SPIR-V at build time

The Android NDK ships `glslc` as a *host* tool (`shader-tools/<host>/glslc`) and ships shaderc only as
source, so there is no on-device compiler to rely on. Compiling at build time also lets `spirv-val`
check the exact bytes that ship. The same GLSL bodies are reused for OpenGL ES with a different
header (`header_es.glsl`: `#version 310 es` and uniforms instead of push constants).
