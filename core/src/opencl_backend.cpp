// SPDX-License-Identifier: Apache-2.0
// OpenCL backend. OpenCL is NOT part of the Android NDK, so nothing is linked: the vendor
// runtime is opened with dlopen at run time and only the small ABI subset below is used
// (values from the Khronos OpenCL 1.2 headers). On Android 12+ the manifest must also declare
// <uses-native-library android:name="libOpenCL.so" android:required="false"/>.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "backend_util.hpp"
#include "hpc/dynlib.hpp"
#include "hpc/log.hpp"

namespace hpc {
namespace {

// ---- minimal OpenCL 1.2 ABI -------------------------------------------------
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bool = uint32_t;
using cl_bitfield = uint64_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1, CL_MEM_WRITE_ONLY = 2, CL_MEM_READ_ONLY = 4, CL_MEM_COPY_HOST_PTR = 1 << 5;
constexpr cl_bitfield CL_QUEUE_PROFILING_ENABLE = 1 << 1;
constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000, CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002, CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004,
                  CL_DEVICE_NAME = 0x102B, CL_DEVICE_VENDOR = 0x102C, CL_DRIVER_VERSION = 0x102D,
                  CL_DEVICE_VERSION = 0x102F, CL_DEVICE_EXTENSIONS = 0x1030, CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035,
                  CL_DEVICE_LOCAL_MEM_SIZE = 0x1023;
constexpr cl_uint CL_PROFILING_COMMAND_START = 0x1282, CL_PROFILING_COMMAND_END = 0x1283, CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_uint CL_KERNEL_WORK_GROUP_SIZE = 0x11B0;
constexpr cl_bitfield CL_DEVICE_TYPE_CPU = 1 << 1, CL_DEVICE_TYPE_GPU = 1 << 2, CL_DEVICE_TYPE_ACCELERATOR = 1 << 3,
                      CL_DEVICE_TYPE_ALL = 0xFFFFFFFFull;

struct ClApi {
  DynLib lib;
  cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*GetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*) = nullptr;
  cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*) = nullptr;
  cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
  cl_context (*CreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*ReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*) = nullptr;
  cl_int (*ReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*) = nullptr;
  cl_int (*ReleaseMemObject)(cl_mem) = nullptr;
  cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
  cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
  cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
  cl_int (*ReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*ReleaseKernel)(cl_kernel) = nullptr;
  cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
  cl_int (*GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
  cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*,
                                 cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*,
                              cl_event*) = nullptr;
  cl_int (*Finish)(cl_command_queue) = nullptr;
  cl_int (*WaitForEvents)(cl_uint, const cl_event*) = nullptr;
  cl_int (*GetEventProfilingInfo)(cl_event, cl_uint, size_t, void*, size_t*) = nullptr;
  cl_int (*ReleaseEvent)(cl_event) = nullptr;
  std::string error;

  bool load() {
    std::string tried;
    if (!lib.open({
#if defined(_WIN32)
            "OpenCL.dll"
#else
            "libOpenCL.so", "libOpenCL.so.1", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",
            "/vendor/lib/libOpenCL.so", "libOpenCL-pixel.so", "/system/lib64/libOpenCL.so",
            "libOpenCL-car.so"
#endif
        },
                  &tried)) {
      error = sanitizeAscii("no OpenCL runtime found (tried: " + tried + ")", 256);
      HPC_LOGW("[CL] %s", error.c_str());
      return false;
    }
#define L(n)                                                                             \
  n = lib.sym<decltype(n)>("cl" #n);                                                     \
  if (!n) {                                                                              \
    error = sanitizeAscii(std::string("missing symbol cl" #n " in ") + lib.path(), 256); \
    HPC_LOGW("[CL] %s", error.c_str());                                                  \
    lib.close();                                                                         \
    return false;                                                                        \
  }
    L(GetPlatformIDs) L(GetPlatformInfo) L(GetDeviceIDs) L(GetDeviceInfo) L(CreateContext) L(ReleaseContext)
    L(CreateCommandQueue) L(ReleaseCommandQueue) L(CreateBuffer) L(ReleaseMemObject) L(CreateProgramWithSource)
    L(BuildProgram) L(GetProgramBuildInfo) L(ReleaseProgram) L(CreateKernel) L(ReleaseKernel) L(SetKernelArg) L(GetKernelWorkGroupInfo)
    L(EnqueueNDRangeKernel) L(EnqueueReadBuffer) L(Finish) L(WaitForEvents) L(GetEventProfilingInfo) L(ReleaseEvent)
#undef L
    HPC_LOGI("[CL] loaded OpenCL runtime from %s", lib.path().c_str());
    return true;
  }
};

// T: 2D tile edge (T*T work-items), WG1: 1D work-group size. Both are set by build options,
// chosen per device AND per kernel from CL_KERNEL_WORK_GROUP_SIZE (it can be lower than the
// device maximum when a kernel needs many registers, e.g. on Mali and Adreno).
const char* kSource = R"CLC(
#ifndef T
#define T 16
#endif
#ifndef WG1
#define WG1 256
#endif
__kernel void sgemm(__global const float* A, __global const float* B, __global float* C,
                    __global const float* bias, uint M, uint K, uint N, uint mode) {
  __local float As[T][T];
  __local float Bs[T][T];
  uint row = get_global_id(1), col = get_global_id(0), lr = get_local_id(1), lc = get_local_id(0);
  float acc = 0.0f;
  uint tiles = (K + T - 1) / T;
  for (uint t = 0; t < tiles; ++t) {
    uint ak = t * T + lc, bk = t * T + lr;
    As[lr][lc] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
    Bs[lr][lc] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < T; ++i) acc += As[lr][i] * Bs[i][lc];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < M && col < N) {
    float v = acc;
    if (mode == 1u) v = fmax(v + bias[col], 0.0f);
    C[row * N + col] = v;
  }
}
__kernel void conv3x3(__global const float* A, __global const float* W, __global float* C, uint M, uint N) {
  uint x = get_global_id(0), y = get_global_id(1);
  if (x >= N || y >= M) return;
  if (x == 0 || y == 0 || x == N - 1 || y == M - 1) { C[y * N + x] = A[y * N + x]; return; }
  float s = 0.0f;
  for (uint dy = 0; dy < 3; ++dy)
    for (uint dx = 0; dx < 3; ++dx)
      s += A[(y + dy - 1) * N + (x + dx - 1)] * W[dy * 3 + dx];
  C[y * N + x] = s;
}
__kernel void reduce_pass(__global const float* A, __global float* C, uint n) {
  __local float sm[WG1];
  uint tid = get_local_id(0);
  uint stride = get_num_groups(0) * WG1;
  float s = 0.0f;
  for (uint i = get_global_id(0); i < n; i += stride) s += A[i];
  sm[tid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint o = WG1 / 2; o > 0; o >>= 1) {
    if (tid < o) sm[tid] += sm[tid + o];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) C[get_group_id(0)] = sm[0];
}
__kernel void triad(__global const float* A, __global const float* B, __global float* C, uint n, float alpha) {
  uint stride = get_global_size(0);
  for (uint i = get_global_id(0); i < n; i += stride) C[i] = A[i] + alpha * B[i];
}
)CLC";

std::string platInfoStr(ClApi& cl, cl_platform_id p, cl_uint what, size_t maxOut = 256) {
  if (!p || !cl.GetPlatformInfo) return "";
  size_t n = 0;
  if (cl.GetPlatformInfo(p, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
  n = std::min<size_t>(n, 4096);
  std::vector<char> buf(n, '\0');
  if (cl.GetPlatformInfo(p, what, n, buf.data(), nullptr) != CL_SUCCESS) return "";
  buf.back() = '\0';
  return sanitizeAscii(buf.data(), maxOut);
}

std::string infoStr(ClApi& cl, cl_device_id d, cl_uint what, size_t maxOut = 256) {
  if (!d || !cl.GetDeviceInfo) return "";
  size_t n = 0;
  if (cl.GetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
  n = std::min<size_t>(n, 4096);
  std::vector<char> buf(n, '\0');
  if (cl.GetDeviceInfo(d, what, n, buf.data(), nullptr) != CL_SUCCESS) return "";
  buf.back() = '\0';
  return sanitizeAscii(buf.data(), maxOut);
}

struct ClDevice {
  std::shared_ptr<ClApi> api;
  cl_device_id dev = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  size_t maxWg = 0;
  cl_ulong localMem = 0;
  ~ClDevice() {
    if (q) api->ReleaseCommandQueue(q);
    if (ctx) api->ReleaseContext(ctx);
  }
  bool init(std::string* err) {
    if (ctx && q) return true;
    if (!dev) {
      if (err) *err = "null cl_device_id";
      return false;
    }
    cl_int e = 0;
    ctx = api->CreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    if (!ctx || e != CL_SUCCESS) {
      ctx = nullptr;
      if (err) *err = "clCreateContext failed: " + std::to_string(e);
      HPC_LOGW("[CL] clCreateContext failed: %d", int(e));
      return false;
    }
    q = api->CreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!q || e != CL_SUCCESS) {
      q = nullptr;
      api->ReleaseContext(ctx);
      ctx = nullptr;
      if (err) *err = "clCreateCommandQueue failed: " + std::to_string(e);
      HPC_LOGW("[CL] clCreateCommandQueue failed: %d", int(e));
      return false;
    }
    HPC_LOGI("[CL] context (%p) and command queue (%p) created", static_cast<void*>(ctx), static_cast<void*>(q));
    return true;
  }
};

class ClExecution final : public Execution {
 public:
  explicit ClExecution(std::shared_ptr<ClDevice> d) : d_(std::move(d)), cl_(*d_->api) {}
  ~ClExecution() override {
    releaseProgram();
    for (auto m : {a_, b_, bias_, partial_, out_}) if (m) cl_.ReleaseMemObject(m);
  }

  bool build(const Problem& p, std::string* err) {
    p_ = &p;
    outCount_ = p.outputCount();
    auto buf = [&](cl_bitfield flags, size_t bytes, void* host) -> cl_mem {
      cl_int e = 0;
      cl_mem m = cl_.CreateBuffer(d_->ctx, flags, std::max<size_t>(16, bytes), host, &e);
      if (!m || e != CL_SUCCESS) {
        if (err) *err = "clCreateBuffer failed: " + std::to_string(e);
        return nullptr;
      }
      return m;
    };
    const cl_bitfield ro = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    const size_t F = sizeof(float);
    // COPY_HOST_PTR needs >= bytes of host memory; pad tiny buffers via a local copy.
    float pad[4] = {0, 0, 0, 0};
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense:
        a_ = buf(ro, size_t(p.m) * p.k * F, const_cast<float*>(p.a.data()));
        b_ = buf(ro, size_t(p.k) * p.n * F, const_cast<float*>(p.b.data()));
        bias_ = p.kernel == Kernel::Dense ? buf(ro, size_t(p.n) * F, const_cast<float*>(p.bias.data())) : buf(ro, 16, pad);
        out_ = buf(CL_MEM_WRITE_ONLY, outCount_ * F, nullptr);
        if (!a_ || !b_ || !bias_ || !out_) return false;
        break;
      case Kernel::Conv3x3:
        a_ = buf(ro, size_t(p.m) * p.n * F, const_cast<float*>(p.a.data()));
        b_ = buf(ro, 9 * F, const_cast<float*>(p.b.data()));
        out_ = buf(CL_MEM_WRITE_ONLY, outCount_ * F, nullptr);
        if (!a_ || !b_ || !out_) return false;
        break;
      case Kernel::Reduce:
        a_ = buf(ro, size_t(p.n) * F, const_cast<float*>(p.a.data()));
        partial_ = buf(CL_MEM_READ_WRITE, kGroups * F, nullptr);
        out_ = buf(CL_MEM_READ_WRITE, F, nullptr);
        if (!a_ || !partial_ || !out_) return false;
        break;
      case Kernel::Triad:
        a_ = buf(ro, size_t(p.n) * F, const_cast<float*>(p.a.data()));
        b_ = buf(ro, size_t(p.n) * F, const_cast<float*>(p.b.data()));
        out_ = buf(CL_MEM_WRITE_ONLY, size_t(p.n) * F, nullptr);
        if (!a_ || !b_ || !out_) return false;
        break;
    }

    // Work-group shapes, largest first; each must fit the device and the compiled kernel.
    struct Shape { uint32_t t, wg1; };
    static const Shape kShapes[] = {{16, 256}, {8, 128}, {8, 64}};
    const bool twoD = p.kernel == Kernel::Sgemm || p.kernel == Kernel::Dense || p.kernel == Kernel::Conv3x3;
    const char* kname = (p.kernel == Kernel::Conv3x3) ? "conv3x3" : (p.kernel == Kernel::Reduce) ? "reduce_pass"
                        : (p.kernel == Kernel::Triad) ? "triad" : "sgemm";
    std::string lastErr = "no work-group shape fits this device";
    bool built = false;
    for (const Shape& sh : kShapes) {
      const size_t need = twoD ? size_t(sh.t) * sh.t : sh.wg1;
      const cl_ulong lmemNeed = (p.kernel == Kernel::Sgemm || p.kernel == Kernel::Dense) ? 2ull * sh.t * sh.t * 4
                                : (p.kernel == Kernel::Reduce) ? cl_ulong(sh.wg1) * 4 : 0;
      if (d_->maxWg && need > d_->maxWg) continue;
      if (d_->localMem && lmemNeed > d_->localMem) continue;
      releaseProgram();
      if (!buildProgram(sh.t, sh.wg1, &lastErr)) continue;
      cl_int e = 0;
      k1_ = cl_.CreateKernel(prog_, kname, &e);
      if (!k1_ || e != CL_SUCCESS) { k1_ = nullptr; lastErr = "clCreateKernel(" + std::string(kname) + ") failed: " + std::to_string(e); continue; }
      size_t kwg = 0;
      if (cl_.GetKernelWorkGroupInfo(k1_, d_->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof kwg, &kwg, nullptr) != CL_SUCCESS) kwg = need;
      if (kwg < need) {
        lastErr = "kernel " + std::string(kname) + " allows only " + std::to_string(kwg) + " work-items per group";
        HPC_LOGI("[CL] %s: CL_KERNEL_WORK_GROUP_SIZE=%zu < %zu, trying a smaller shape", kname, kwg, need);
        continue;
      }
      if (p.kernel == Kernel::Reduce) {
        k2_ = cl_.CreateKernel(prog_, kname, &e);
        if (!k2_ || e != CL_SUCCESS) { k2_ = nullptr; lastErr = "clCreateKernel(reduce pass 2) failed: " + std::to_string(e); continue; }
      }
      tile_ = sh.t;
      wg1_ = sh.wg1;
      built = true;
      break;
    }
    if (!built) {
      if (err) *err = lastErr;
      return false;
    }
    HPC_LOGI("[CL] %s: work-group shape %s", kname,
             twoD ? (std::to_string(tile_) + "x" + std::to_string(tile_)).c_str() : std::to_string(wg1_).c_str());

    auto arg = [&](cl_kernel k, cl_uint i, size_t sz, const void* v) { return cl_.SetKernelArg(k, i, sz, v) == CL_SUCCESS; };
    auto roundUp = [](size_t v, size_t m) { return (v + m - 1) / m * m; };
    bool ok = true;
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense: {
        const cl_uint M = p.m, K = p.k, N = p.n, mode = p.kernel == Kernel::Dense ? 1u : 0u;
        ok = arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, sizeof(cl_mem), &bias_) && arg(k1_, 4, 4, &M) && arg(k1_, 5, 4, &K) && arg(k1_, 6, 4, &N) &&
             arg(k1_, 7, 4, &mode);
        gws_[0] = roundUp(N, tile_); gws_[1] = roundUp(M, tile_); lws_[0] = lws_[1] = tile_; dims_ = 2;
        break;
      }
      case Kernel::Conv3x3: {
        const cl_uint M = p.m, N = p.n;
        ok = arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, 4, &M) && arg(k1_, 4, 4, &N);
        gws_[0] = roundUp(N, tile_); gws_[1] = roundUp(M, tile_); lws_[0] = lws_[1] = tile_; dims_ = 2;
        break;
      }
      case Kernel::Reduce: {
        const cl_uint n = p.n;
        groups_ = std::min<uint32_t>(kGroups, std::max<uint32_t>(1, (p.n + wg1_ - 1) / wg1_));
        const cl_uint n2 = groups_;
        ok = arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &partial_) && arg(k1_, 2, 4, &n) &&
             arg(k2_, 0, sizeof(cl_mem), &partial_) && arg(k2_, 1, sizeof(cl_mem), &out_) && arg(k2_, 2, 4, &n2);
        gws_[0] = size_t(groups_) * wg1_; lws_[0] = wg1_; dims_ = 1;
        break;
      }
      case Kernel::Triad: {
        const cl_uint n = p.n;
        const float alpha = p.alpha;
        ok = arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, 4, &n) && arg(k1_, 4, 4, &alpha);
        gws_[0] = size_t(std::min<uint32_t>(8192, std::max<uint32_t>(1, (p.n + wg1_ - 1) / wg1_))) * wg1_; lws_[0] = wg1_; dims_ = 1;
        break;
      }
    }
    if (!ok) {
      if (err) *err = "failed to set OpenCL kernel arguments";
      return false;
    }
    return true;
  }

  void releaseProgram() {
    if (k1_) cl_.ReleaseKernel(k1_);
    if (k2_) cl_.ReleaseKernel(k2_);
    if (prog_) cl_.ReleaseProgram(prog_);
    k1_ = k2_ = nullptr;
    prog_ = nullptr;
  }

  bool buildProgram(uint32_t tile, uint32_t wg1, std::string* err) {
    cl_int e = 0;
    const char* src = kSource;
    prog_ = cl_.CreateProgramWithSource(d_->ctx, 1, &src, nullptr, &e);
    if (!prog_ || e != CL_SUCCESS) {
      prog_ = nullptr;
      if (err) *err = "clCreateProgramWithSource failed: " + std::to_string(e);
      return false;
    }
    const std::string opts = "-DT=" + std::to_string(tile) + " -DWG1=" + std::to_string(wg1);
    e = cl_.BuildProgram(prog_, 1, &d_->dev, opts.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
      size_t n = 0;
      cl_.GetProgramBuildInfo(prog_, d_->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
      n = std::min<size_t>(n, 4096);
      std::vector<char> log(n ? n : 1, '\0');
      if (n) cl_.GetProgramBuildInfo(prog_, d_->dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
      log.back() = '\0';
      const std::string cleanLog = sanitizeAscii(log.data(), 256);
      HPC_LOGW("[CL] clBuildProgram(%s) failed (%d): %s", opts.c_str(), int(e), cleanLog.c_str());
      if (err) *err = "OpenCL build failed (" + std::to_string(e) + "): " + cleanLog;
      return false;
    }
    return true;
  }

  bool run(double* deviceMs, std::string* err) override {
    if (deviceMs) *deviceMs = -1;
    cl_event e1 = nullptr, e2 = nullptr;
    cl_int r = cl_.EnqueueNDRangeKernel(d_->q, k1_, dims_, nullptr, gws_, lws_, 0, nullptr, &e1);
    if (r == CL_SUCCESS && k2_) {
      const size_t g2 = wg1_, l2 = wg1_;
      r = cl_.EnqueueNDRangeKernel(d_->q, k2_, 1, nullptr, &g2, &l2, 0, nullptr, &e2);
    }
    if (r != CL_SUCCESS) {
      if (e1) cl_.ReleaseEvent(e1);
      if (e2) cl_.ReleaseEvent(e2);
      if (err) *err = "clEnqueueNDRangeKernel failed: " + std::to_string(r);
      return false;
    }
    r = cl_.Finish(d_->q);
    if (r != CL_SUCCESS) {
      if (e1) cl_.ReleaseEvent(e1);
      if (e2) cl_.ReleaseEvent(e2);
      if (err) *err = "clFinish failed: " + std::to_string(r);
      return false;
    }
    cl_event last = e2 ? e2 : e1;
    cl_ulong t0 = 0, t1 = 0;
    if (cl_.GetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr) == CL_SUCCESS &&
        cl_.GetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr) == CL_SUCCESS && t1 >= t0 && deviceMs)
      *deviceMs = double(t1 - t0) * 1e-6;
    if (e1) cl_.ReleaseEvent(e1);
    if (e2) cl_.ReleaseEvent(e2);
    return true;
  }

  bool read(std::vector<float>& out, std::string* err) override {
    out.resize(outCount_);
    cl_int r = cl_.EnqueueReadBuffer(d_->q, out_, CL_TRUE, 0, outCount_ * sizeof(float), out.data(), 0, nullptr, nullptr);
    if (r != CL_SUCCESS) {
      if (err) *err = "clEnqueueReadBuffer failed: " + std::to_string(r);
      return false;
    }
    return true;
  }

 private:
  static constexpr uint32_t kGroups = 256;
  std::shared_ptr<ClDevice> d_;
  ClApi& cl_;
  const Problem* p_ = nullptr;
  size_t outCount_ = 0;
  cl_mem a_ = nullptr, b_ = nullptr, bias_ = nullptr, partial_ = nullptr, out_ = nullptr;
  cl_program prog_ = nullptr;
  cl_kernel k1_ = nullptr, k2_ = nullptr;
  uint32_t tile_ = 16, wg1_ = 256;
  size_t gws_[2] = {1, 1}, lws_[2] = {1, 1};
  cl_uint dims_ = 1;
  uint32_t groups_ = 1;
};

class OpenCLBackend final : public Backend {
 public:
  OpenCLBackend(std::shared_ptr<ClApi> api, cl_platform_id plat, cl_device_id dev, int index) {
    d_ = std::make_shared<ClDevice>();
    d_->api = api;
    d_->dev = dev;
    ClApi& cl = *api;
    const std::string pname = platInfoStr(cl, plat, CL_PLATFORM_NAME, 256);
    cl_bitfield type = 0;
    if (cl.GetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof type, &type, nullptr) != CL_SUCCESS) type = 0;
    cl_uint cu = 0;
    if (cl.GetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, nullptr) != CL_SUCCESS) cu = 0;
    size_t wg = 0;
    if (cl.GetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr) != CL_SUCCESS) wg = 0;
    cl_ulong lmem = 0;
    if (cl.GetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof lmem, &lmem, nullptr) != CL_SUCCESS) lmem = 0;
    cl_bool unified = 0;
    if (cl.GetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, nullptr) != CL_SUCCESS) unified = 0;
    d_->maxWg = wg;
    // Test hook: cap the work-group size to exercise the smaller-shape fallback on any runtime.
    if (const char* cap = std::getenv("HPC_TEST_CL_MAX_WG")) {
      const unsigned long c = std::strtoul(cap, nullptr, 10);
      if (c > 0 && c < d_->maxWg) d_->maxWg = c;
    }
    d_->localMem = lmem;

    const std::string devVer = infoStr(cl, dev, CL_DEVICE_VERSION, 256);
    const std::string exts = infoStr(cl, dev, CL_DEVICE_EXTENSIONS, 4096);
    info_.id = "opencl:" + std::to_string(index);
    info_.api = sanitizeAscii("OpenCL (" + devVer + ")", 256);
    info_.device = infoStr(cl, dev, CL_DEVICE_NAME, 256);
    info_.vendor = infoStr(cl, dev, CL_DEVICE_VENDOR, 256);
    info_.driver = infoStr(cl, dev, CL_DRIVER_VERSION, 256);
    info_.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
    info_.available = dev != nullptr && wg >= 64 && lmem >= 512;
    if (!info_.available) info_.note = "needs work-group size >= 64 and >= 512 B local memory";
    auto& dt = info_.detail;
    dt["platform"] = pname;
    dt["device_type"] = (type & CL_DEVICE_TYPE_GPU) ? "gpu" : (type & CL_DEVICE_TYPE_CPU) ? "cpu" : (type & CL_DEVICE_TYPE_ACCELERATOR) ? "accelerator" : "other";
    info_.deviceClass = (type & CL_DEVICE_TYPE_GPU) ? "gpu" : (type & CL_DEVICE_TYPE_CPU) ? "cpu" : (type & CL_DEVICE_TYPE_ACCELERATOR) ? "npu" : "other";
    dt["compute_units"] = std::to_string(cu);
    dt["max_work_group_size"] = std::to_string(wg);
    dt["local_mem_bytes"] = std::to_string(lmem);
    dt["host_unified_memory"] = unified ? "yes" : "no";
    dt["fp64"] = exts.find("cl_khr_fp64") != std::string::npos ? "yes" : "no";
    dt["runtime_library"] = sanitizeAscii(api->lib.path(), 256);
    HPC_LOGI("[CL] device %s: name='%s' vendor='%s' driver='%s' type=%s cu=%u maxWg=%zu localMem=%llu available=%d",
             info_.id.c_str(), info_.device.c_str(), info_.vendor.c_str(), info_.driver.c_str(),
             dt["device_type"].c_str(), unsigned(cu), wg, static_cast<unsigned long long>(lmem),
             info_.available ? 1 : 0);
  }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions&, std::string* err) override {
    if (!d_->init(err)) return nullptr;
    std::unique_ptr<ClExecution> e(new ClExecution(d_));
    if (!e->build(p, err)) return nullptr;
    return e;
  }

 private:
  std::shared_ptr<ClDevice> d_;
  BackendInfo info_;
};

}  // namespace

void discoverOpenCL(std::vector<std::unique_ptr<Backend>>& out) {
  auto api = std::make_shared<ClApi>();
  auto stub = [&](const std::string& why) {
    BackendInfo i;
    i.id = "opencl";
    i.api = "OpenCL";
    i.note = sanitizeAscii(why, 256);
    HPC_LOGW("[CL] unavailable: %s", i.note.c_str());
    out.emplace_back(new StubBackend(i));
  };
  if (!api->load()) return stub(api->error);
  cl_uint np = 0;
  cl_int r = api->GetPlatformIDs(0, nullptr, &np);
  HPC_LOGI("[CL] clGetPlatformIDs(count) -> rc=%d count=%u", int(r), unsigned(np));
  if (r != CL_SUCCESS || np == 0) {
    return stub("OpenCL runtime loaded (" + sanitizeAscii(api->lib.path(), 160) +
                ") but reports no platforms (code " + std::to_string(r) + ")");
  }
  if (np > 64) {
    HPC_LOGW("[CL] clamping unreasonable platform count %u -> 64", unsigned(np));
    np = 64;
  }
  std::vector<cl_platform_id> plats(np, nullptr);
  r = api->GetPlatformIDs(np, plats.data(), nullptr);
  if (r != CL_SUCCESS) {
    return stub("clGetPlatformIDs(list) failed (code " + std::to_string(r) + ")");
  }
  int idx = 0;
  for (cl_uint pi = 0; pi < np; ++pi) {
    cl_platform_id pl = plats[pi];
    if (!pl) continue;
    const std::string pname = platInfoStr(*api, pl, CL_PLATFORM_NAME, 256);
    cl_uint nd = 0;
    cl_int dr = api->GetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    HPC_LOGI("[CL] platform[%u] '%s': clGetDeviceIDs(count) -> rc=%d count=%u",
             unsigned(pi), pname.c_str(), int(dr), unsigned(nd));
    if (dr != CL_SUCCESS || nd == 0) continue;
    if (nd > 64) {
      HPC_LOGW("[CL] clamping unreasonable device count %u -> 64", unsigned(nd));
      nd = 64;
    }
    std::vector<cl_device_id> devs(nd, nullptr);
    dr = api->GetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    if (dr != CL_SUCCESS) {
      HPC_LOGW("[CL] platform[%u] clGetDeviceIDs(list) failed: %d", unsigned(pi), int(dr));
      continue;
    }
    for (auto d : devs) {
      if (!d) continue;
      out.emplace_back(new OpenCLBackend(api, pl, d, idx++));
    }
  }
  if (idx == 0) stub("OpenCL platforms present but no devices");
}

}  // namespace hpc
