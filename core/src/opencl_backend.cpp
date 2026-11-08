// SPDX-License-Identifier: Apache-2.0
// OpenCL backend. OpenCL is NOT part of the Android NDK, so nothing is linked: the vendor
// runtime is opened with dlopen at run time and only the small ABI subset below is used
// (values from the Khronos OpenCL 1.2 headers). On Android 12+ the manifest must also declare
// <uses-native-library android:name="libOpenCL.so" android:required="false"/>.
#include <algorithm>
#include <cstring>

#include "backend_util.hpp"
#include "hpc/dynlib.hpp"

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
            "libOpenCL-car.so", "libGLES_mali.so"  // last resort: some Mali drivers export OpenCL from here
#endif
        },
                  &tried)) {
      error = "no OpenCL runtime found (tried: " + tried + ")";
      return false;
    }
#define L(n)                                                          \
  n = lib.sym<decltype(n)>("cl" #n);                                  \
  if (!n) {                                                           \
    error = std::string("missing symbol cl" #n " in ") + lib.path();  \
    return false;                                                     \
  }
    L(GetPlatformIDs) L(GetPlatformInfo) L(GetDeviceIDs) L(GetDeviceInfo) L(CreateContext) L(ReleaseContext)
    L(CreateCommandQueue) L(ReleaseCommandQueue) L(CreateBuffer) L(ReleaseMemObject) L(CreateProgramWithSource)
    L(BuildProgram) L(GetProgramBuildInfo) L(ReleaseProgram) L(CreateKernel) L(ReleaseKernel) L(SetKernelArg)
    L(EnqueueNDRangeKernel) L(EnqueueReadBuffer) L(Finish) L(WaitForEvents) L(GetEventProfilingInfo) L(ReleaseEvent)
#undef L
    return true;
  }
};

const char* kSource = R"CLC(
#define T 16
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
  __local float sm[256];
  uint tid = get_local_id(0);
  uint stride = get_num_groups(0) * 256;
  float s = 0.0f;
  for (uint i = get_global_id(0); i < n; i += stride) s += A[i];
  sm[tid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint o = 128; o > 0; o >>= 1) {
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

std::string infoStr(ClApi& cl, cl_device_id d, cl_uint what) {
  size_t n = 0;
  if (cl.GetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
  std::string s(n, '\0');
  cl.GetDeviceInfo(d, what, n, &s[0], nullptr);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

struct ClDevice {
  std::shared_ptr<ClApi> api;
  cl_device_id dev = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  size_t maxWg = 0;
  ~ClDevice() {
    if (q) api->ReleaseCommandQueue(q);
    if (ctx) api->ReleaseContext(ctx);
  }
  bool init(std::string* err) {
    if (ctx) return true;
    cl_int e = 0;
    ctx = api->CreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    if (!ctx || e != CL_SUCCESS) {
      ctx = nullptr;
      if (err) *err = "clCreateContext failed: " + std::to_string(e);
      return false;
    }
    q = api->CreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    if (!q || e != CL_SUCCESS) {
      q = nullptr;
      if (err) *err = "clCreateCommandQueue failed: " + std::to_string(e);
      return false;
    }
    return true;
  }
};

class ClExecution final : public Execution {
 public:
  explicit ClExecution(std::shared_ptr<ClDevice> d) : d_(std::move(d)), cl_(*d_->api) {}
  ~ClExecution() override {
    for (auto k : kernels_) if (k) cl_.ReleaseKernel(k);
    if (prog_) cl_.ReleaseProgram(prog_);
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

    cl_int e = 0;
    const char* src = kSource;
    prog_ = cl_.CreateProgramWithSource(d_->ctx, 1, &src, nullptr, &e);
    if (!prog_ || e != CL_SUCCESS) {
      prog_ = nullptr;
      if (err) *err = "clCreateProgramWithSource failed: " + std::to_string(e);
      return false;
    }
    e = cl_.BuildProgram(prog_, 1, &d_->dev, "", nullptr, nullptr);
    if (e != CL_SUCCESS) {
      size_t n = 0;
      cl_.GetProgramBuildInfo(prog_, d_->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
      std::string log(n, '\0');
      if (n) cl_.GetProgramBuildInfo(prog_, d_->dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
      if (err) *err = "OpenCL build failed (" + std::to_string(e) + "): " + log.substr(0, 400);
      return false;
    }
    auto mk = [&](const char* name) {
      cl_kernel k = cl_.CreateKernel(prog_, name, &e);
      kernels_.push_back(k);
      return k;
    };
    auto arg = [&](cl_kernel k, cl_uint i, size_t sz, const void* v) { return cl_.SetKernelArg(k, i, sz, v) == CL_SUCCESS; };
    bool ok = true;
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense: {
        k1_ = mk("sgemm");
        const cl_uint M = p.m, K = p.k, N = p.n, mode = p.kernel == Kernel::Dense ? 1u : 0u;
        ok = k1_ && arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, sizeof(cl_mem), &bias_) && arg(k1_, 4, 4, &M) && arg(k1_, 5, 4, &K) && arg(k1_, 6, 4, &N) &&
             arg(k1_, 7, 4, &mode);
        gws_[0] = (N + 15) / 16 * 16; gws_[1] = (M + 15) / 16 * 16; lws_[0] = lws_[1] = 16; dims_ = 2;
        break;
      }
      case Kernel::Conv3x3: {
        k1_ = mk("conv3x3");
        const cl_uint M = p.m, N = p.n;
        ok = k1_ && arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, 4, &M) && arg(k1_, 4, 4, &N);
        gws_[0] = (N + 15) / 16 * 16; gws_[1] = (M + 15) / 16 * 16; lws_[0] = lws_[1] = 16; dims_ = 2;
        break;
      }
      case Kernel::Reduce: {
        k1_ = mk("reduce_pass");
        k2_ = mk("reduce_pass");
        const cl_uint n = p.n;
        groups_ = std::min<uint32_t>(kGroups, std::max<uint32_t>(1, (p.n + 255) / 256));
        const cl_uint n2 = groups_;
        ok = k1_ && k2_ && arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &partial_) && arg(k1_, 2, 4, &n) &&
             arg(k2_, 0, sizeof(cl_mem), &partial_) && arg(k2_, 1, sizeof(cl_mem), &out_) && arg(k2_, 2, 4, &n2);
        gws_[0] = size_t(groups_) * 256; lws_[0] = 256; dims_ = 1;
        break;
      }
      case Kernel::Triad: {
        k1_ = mk("triad");
        const cl_uint n = p.n;
        const float alpha = p.alpha;
        ok = k1_ && arg(k1_, 0, sizeof(cl_mem), &a_) && arg(k1_, 1, sizeof(cl_mem), &b_) && arg(k1_, 2, sizeof(cl_mem), &out_) &&
             arg(k1_, 3, 4, &n) && arg(k1_, 4, 4, &alpha);
        gws_[0] = size_t(std::min<uint32_t>(8192, std::max<uint32_t>(1, (p.n + 255) / 256))) * 256; lws_[0] = 256; dims_ = 1;
        break;
      }
    }
    if (!ok) {
      if (err) *err = "failed to create/configure OpenCL kernel";
      return false;
    }
    return true;
  }

  bool run(double* deviceMs, std::string* err) override {
    if (deviceMs) *deviceMs = -1;
    cl_event e1 = nullptr, e2 = nullptr;
    cl_int r = cl_.EnqueueNDRangeKernel(d_->q, k1_, dims_, nullptr, gws_, lws_, 0, nullptr, &e1);
    if (r == CL_SUCCESS && k2_) {
      const size_t g2 = 256, l2 = 256;
      r = cl_.EnqueueNDRangeKernel(d_->q, k2_, 1, nullptr, &g2, &l2, 0, nullptr, &e2);
    }
    if (r != CL_SUCCESS) {
      if (err) *err = "clEnqueueNDRangeKernel failed: " + std::to_string(r);
      return false;
    }
    cl_.Finish(d_->q);
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
  std::vector<cl_kernel> kernels_;
  cl_kernel k1_ = nullptr, k2_ = nullptr;
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
    size_t n = 0;
    cl.GetPlatformInfo(plat, CL_PLATFORM_NAME, 0, nullptr, &n);
    std::string pname(n, '\0');
    if (n) cl.GetPlatformInfo(plat, CL_PLATFORM_NAME, n, &pname[0], nullptr);
    while (!pname.empty() && pname.back() == '\0') pname.pop_back();
    cl_bitfield type = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof type, &type, nullptr);
    cl_uint cu = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, nullptr);
    size_t wg = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr);
    cl_ulong lmem = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof lmem, &lmem, nullptr);
    cl_bool unified = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, nullptr);
    d_->maxWg = wg;

    info_.id = "opencl:" + std::to_string(index);
    info_.api = "OpenCL (" + infoStr(cl, dev, CL_DEVICE_VERSION) + ")";
    info_.device = infoStr(cl, dev, CL_DEVICE_NAME);
    info_.vendor = infoStr(cl, dev, CL_DEVICE_VENDOR);
    info_.driver = infoStr(cl, dev, CL_DRIVER_VERSION);
    info_.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
    info_.available = wg >= 256 && lmem >= 4096;
    if (!info_.available) info_.note = "needs work-group size >= 256 and >= 4 KiB local memory";
    auto& dt = info_.detail;
    dt["platform"] = pname;
    dt["device_type"] = (type & CL_DEVICE_TYPE_GPU) ? "gpu" : (type & CL_DEVICE_TYPE_CPU) ? "cpu" : (type & CL_DEVICE_TYPE_ACCELERATOR) ? "accelerator" : "other";
    dt["compute_units"] = std::to_string(cu);
    dt["max_work_group_size"] = std::to_string(wg);
    dt["local_mem_bytes"] = std::to_string(lmem);
    dt["host_unified_memory"] = unified ? "yes" : "no";
    dt["fp64"] = infoStr(cl, dev, CL_DEVICE_EXTENSIONS).find("cl_khr_fp64") != std::string::npos ? "yes" : "no";
    dt["runtime_library"] = api->lib.path();
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
    i.note = why;
    out.emplace_back(new StubBackend(i));
  };
  if (!api->load()) return stub(api->error);
  cl_uint np = 0;
  cl_int r = api->GetPlatformIDs(0, nullptr, &np);
  if (r != CL_SUCCESS || np == 0) return stub("OpenCL runtime loaded (" + api->lib.path() + ") but reports no platforms (code " + std::to_string(r) + ")");
  std::vector<cl_platform_id> plats(np);
  api->GetPlatformIDs(np, plats.data(), nullptr);
  int idx = 0;
  for (auto pl : plats) {
    cl_uint nd = 0;
    if (api->GetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
    std::vector<cl_device_id> devs(nd);
    api->GetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    for (auto d : devs) out.emplace_back(new OpenCLBackend(api, pl, d, idx++));
  }
  if (idx == 0) stub("OpenCL platforms present but no devices");
}

}  // namespace hpc
