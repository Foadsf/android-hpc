// SPDX-License-Identifier: Apache-2.0
// SYCL-style backend. The kernels are written with queue.parallel_for(range, lambda), the
// subset of SYCL 2020 that hpc::sycl_shim implements, and run on the CPU thread pool.
// This is a stand-in, NOT real SYCL: BackendInfo::genuine is false and the note says so.
#include <algorithm>

#include "hpc/backend.hpp"
#include "hpc/sycl_shim.hpp"
#include "cpu_pool.hpp"

namespace hpc {
namespace {
namespace sycl = hpc::sycl_shim;

class SyclExecution final : public Execution {
 public:
  SyclExecution(const Problem& p, ThreadPool& pool) : p_(p), q_(pool), out_(p.outputCount()) {}
  bool run(double* deviceMs, std::string*) override {
    if (deviceMs) *deviceMs = -1;
    const uint32_t M = p_.m, K = p_.k, N = p_.n;
    const float *A = p_.a.data(), *B = p_.b.data(), *bias = p_.bias.data();
    float* C = out_.data();
    switch (p_.kernel) {
      case Kernel::Sgemm:
        q_.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> it) {
          const size_t i = it[0], j = it[1];
          float s = 0;
          for (uint32_t kk = 0; kk < K; ++kk) s += A[i * K + kk] * B[size_t(kk) * N + j];
          C[i * N + j] = s;
        });
        break;
      case Kernel::Dense:
        q_.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> it) {
          const size_t i = it[0], j = it[1];
          float s = 0;
          for (uint32_t kk = 0; kk < K; ++kk) s += A[i * K + kk] * B[size_t(kk) * N + j];
          C[i * N + j] = std::max(0.0f, s + bias[j]);
        });
        break;
      case Kernel::Conv3x3:
        q_.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> it) {
          const size_t y = it[0], x = it[1];
          if (y == 0 || x == 0 || y == M - 1 || x == N - 1) {
            C[y * N + x] = A[y * N + x];
            return;
          }
          float s = 0;
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) s += A[(y + dy - 1) * N + (x + dx - 1)] * B[dy * 3 + dx];
          C[y * N + x] = s;
        });
        break;
      case Kernel::Reduce: {
        constexpr uint32_t kChunk = 1u << 16;
        const uint32_t n = p_.n, chunks = (n + kChunk - 1) / kChunk;
        partial_.assign(chunks, 0.0);
        double* P = partial_.data();
        q_.parallel_for(sycl::range<1>(chunks), [=](sycl::id<1> it) {
          const uint32_t b = uint32_t(it[0]) * kChunk, e = std::min(n, b + kChunk);
          double s = 0;
          for (uint32_t i = b; i < e; ++i) s += A[i];
          P[it[0]] = s;
        });
        double t = 0;
        for (double d : partial_) t += d;
        C[0] = float(t);
        break;
      }
      case Kernel::Triad: {
        const float alpha = p_.alpha;
        q_.parallel_for(sycl::range<1>(p_.n), [=](sycl::id<1> it) { C[it[0]] = A[it[0]] + alpha * B[it[0]]; });
        break;
      }
    }
    q_.wait();
    return true;
  }
  bool read(std::vector<float>& out, std::string*) override {
    out = out_;
    return true;
  }

 private:
  const Problem& p_;
  sycl::queue q_;
  std::vector<float> out_;
  std::vector<double> partial_;
};

class SyclBackend final : public Backend {
 public:
  SyclBackend() {
    info_.id = "sycl-shim";
    info_.api = "SYCL-style (NOT real SYCL)";
    info_.device = "CPU thread pool";
    info_.vendor = "-";
    info_.driver = "hpc::sycl_shim";
    info_.deviceClass = "cpu";
    info_.available = true;
    info_.genuine = false;
    info_.note =
        "API-compatible stand-in running on the CPU thread pool. Not a SYCL implementation: "
        "no device compiler, nd_range, local memory or accessors. Real SYCL (AdaptiveCpp) is not "
        "packaged for Android.";
    info_.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
  }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions&, std::string*) override {
    return std::unique_ptr<Execution>(new SyclExecution(p, sharedPool(false, 0)));
  }

 private:
  BackendInfo info_;
};
}  // namespace

std::unique_ptr<Backend> makeSyclShimBackend() { return std::unique_ptr<Backend>(new SyclBackend()); }

}  // namespace hpc
