// SPDX-License-Identifier: Apache-2.0
// Multi-threaded CPU backend. Portable C++17: register-tiled GEMM that relies on the
// compiler's auto-vectoriser (NEON on arm64, SSE/AVX on x86). No hand-written
// intrinsics, so every code path is exercised by the host test-suite.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include "hpc/backend.hpp"
#include "hpc/cpu.hpp"
#include "cpu_pool.hpp"

namespace hpc {

// ---- shared pool ------------------------------------------------------------
namespace {
std::mutex gPoolMu;
std::unique_ptr<ThreadPool> gPool;
bool gPoolBig = false;
unsigned gPoolReserve = 0;
int gPoolPin = -1;
}  // namespace

ThreadPool& sharedPool(bool bigOnly, unsigned reserve, int pinMode) {
  std::lock_guard<std::mutex> lk(gPoolMu);
  if (!gPool || gPoolBig != bigOnly || gPoolReserve != reserve || gPoolPin != pinMode) {
    gPool.reset();  // join old workers before creating new ones
    const CpuInfo ci = probeCpu();
    const auto& cores = bigOnly ? ci.bigCores : ci.cores;
    const unsigned workers = cores.size() > reserve ? unsigned(cores.size()) - reserve : 1u;
    const bool pin = pinMode < 0 ? (ci.heterogeneous || bigOnly) : pinMode > 0;
    std::vector<int> pinTo;
    if (pin) pinTo.assign(cores.begin(), cores.begin() + workers);
    gPool.reset(new ThreadPool(workers, pinTo));
    gPoolBig = bigOnly;
    gPoolReserve = reserve;
    gPoolPin = pinMode;
  }
  return *gPool;
}

namespace {

constexpr int MRmax = 4;  // rows per micro-tile
constexpr int NR = 8;     // cols per micro-tile (one 8-wide panel)

template <int MR>
inline void microKernel(const float* A, size_t lda, const float* Bp, uint32_t K, float out[MRmax][NR]) {
  float c[MR][NR];
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < NR; ++j) c[r][j] = 0.0f;
  for (uint32_t kk = 0; kk < K; ++kk) {
    const float* b = Bp + size_t(kk) * NR;
    for (int r = 0; r < MR; ++r) {
      const float a = A[size_t(r) * lda + kk];
      for (int j = 0; j < NR; ++j) c[r][j] += a * b[j];
    }
  }
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < NR; ++j) out[r][j] = c[r][j];
}

void gemm(ThreadPool& pool, const Problem& p, std::vector<float>& C, std::vector<float>& Bp) {
  const uint32_t M = p.m, K = p.k, N = p.n;
  const uint32_t panels = (N + NR - 1) / NR;
  Bp.resize(size_t(panels) * K * NR);
  // Pack B into column panels so the micro-kernel streams it linearly. Counted in timing.
  pool.parallelFor(panels, 1, [&](uint32_t p0, uint32_t p1) {
    for (uint32_t pn = p0; pn < p1; ++pn)
      for (uint32_t kk = 0; kk < K; ++kk)
        for (int j = 0; j < NR; ++j) {
          const uint32_t col = pn * NR + j;
          Bp[(size_t(pn) * K + kk) * NR + j] = col < N ? p.b[size_t(kk) * N + col] : 0.0f;
        }
  });
  const bool dense = p.kernel == Kernel::Dense;
  const uint32_t rowBlocks = (M + MRmax - 1) / MRmax;
  const uint32_t grain = std::max(1u, rowBlocks / (pool.size() * 8));
  pool.parallelFor(rowBlocks, grain, [&](uint32_t b0, uint32_t b1) {
    for (uint32_t ib = b0; ib < b1; ++ib) {
      const uint32_t i0 = ib * MRmax;
      const int mr = int(std::min<uint32_t>(MRmax, M - i0));
      const float* A = &p.a[size_t(i0) * K];
      for (uint32_t pn = 0; pn < panels; ++pn) {
        float t[MRmax][NR];
        const float* panel = &Bp[size_t(pn) * K * NR];
        switch (mr) {
          case 4: microKernel<4>(A, K, panel, K, t); break;
          case 3: microKernel<3>(A, K, panel, K, t); break;
          case 2: microKernel<2>(A, K, panel, K, t); break;
          default: microKernel<1>(A, K, panel, K, t); break;
        }
        const uint32_t j0 = pn * NR;
        const int nr = int(std::min<uint32_t>(NR, N - j0));
        for (int r = 0; r < mr; ++r)
          for (int j = 0; j < nr; ++j) {
            float v = t[r][j];
            if (dense) v = std::max(0.0f, v + p.bias[j0 + j]);
            C[size_t(i0 + r) * N + j0 + j] = v;
          }
      }
    }
  });
}

void conv3x3(ThreadPool& pool, const Problem& p, std::vector<float>& C) {
  const uint32_t H = p.m, W = p.n;
  const float* w = p.b.data();
  pool.parallelFor(H, 8, [&](uint32_t y0, uint32_t y1) {
    for (uint32_t y = y0; y < y1; ++y) {
      float* o = &C[size_t(y) * W];
      const float* in = &p.a[size_t(y) * W];
      if (y == 0 || y == H - 1) {
        std::memcpy(o, in, size_t(W) * sizeof(float));
        continue;
      }
      o[0] = in[0];
      o[W - 1] = in[W - 1];
      const float* r0 = in - W;
      const float* r1 = in;
      const float* r2 = in + W;
      for (uint32_t x = 1; x + 1 < W; ++x)
        o[x] = r0[x - 1] * w[0] + r0[x] * w[1] + r0[x + 1] * w[2] + r1[x - 1] * w[3] +
               r1[x] * w[4] + r1[x + 1] * w[5] + r2[x - 1] * w[6] + r2[x] * w[7] + r2[x + 1] * w[8];
    }
  });
}

void reduceSum(ThreadPool& pool, const Problem& p, std::vector<float>& C, std::vector<double>& partials) {
  constexpr uint32_t kChunk = 1u << 16;
  const uint32_t chunks = (p.n + kChunk - 1) / kChunk;
  partials.assign(chunks, 0.0);
  pool.parallelFor(chunks, 1, [&](uint32_t c0, uint32_t c1) {
    for (uint32_t c = c0; c < c1; ++c) {
      const uint32_t b = c * kChunk, e = std::min(p.n, b + kChunk);
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      uint32_t i = b;
      for (; i + 8 <= e; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += p.a[i + j];
      double s = 0;
      for (int j = 0; j < 8; ++j) s += acc[j];
      for (; i < e; ++i) s += p.a[i];
      partials[c] = s;
    }
  });
  double total = 0;
  for (double d : partials) total += d;
  C[0] = float(total);
}

void triad(ThreadPool& pool, const Problem& p, std::vector<float>& C) {
  constexpr uint32_t kChunk = 1u << 16;
  const float alpha = p.alpha;
  pool.parallelFor((p.n + kChunk - 1) / kChunk, 1, [&](uint32_t c0, uint32_t c1) {
    for (uint32_t c = c0; c < c1; ++c) {
      const uint32_t b = c * kChunk, e = std::min(p.n, b + kChunk);
      for (uint32_t i = b; i < e; ++i) C[i] = p.a[i] + alpha * p.b[i];
    }
  });
}

class CpuExecution final : public Execution {
 public:
  CpuExecution(const Problem& p, ThreadPool& pool) : p_(p), pool_(pool), out_(p.outputCount()) {}
  bool run(double* deviceMs, std::string*) override {
    if (deviceMs) *deviceMs = -1;
    switch (p_.kernel) {
      case Kernel::Sgemm:
      case Kernel::Dense: gemm(pool_, p_, out_, bp_); break;
      case Kernel::Conv3x3: conv3x3(pool_, p_, out_); break;
      case Kernel::Reduce: reduceSum(pool_, p_, out_, partials_); break;
      case Kernel::Triad: triad(pool_, p_, out_); break;
    }
    return true;
  }
  bool read(std::vector<float>& out, std::string*) override {
    out = out_;
    return true;
  }

 private:
  const Problem& p_;  // must outlive this object (documented in Backend::prepare callers)
  ThreadPool& pool_;
  std::vector<float> out_, bp_;
  std::vector<double> partials_;
};

class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(const DiscoverOptions& opt) : big_(opt.cpuBigCoresOnly), reserve_(opt.cpuReserveCores) {
    const CpuInfo ci = probeCpu();
    const auto& cores = big_ ? ci.bigCores : ci.cores;
    const size_t used = cores.size() > reserve_ ? cores.size() - reserve_ : 1;
    info_.id = "cpu";
    info_.api = "CPU (C++17 threads)";
    info_.device = ci.arch + ", " + std::to_string(used) + " of " +
                   std::to_string(ci.logicalCores) + " cores used";
    info_.vendor = ci.arch;
    info_.driver = "portable C++17, auto-vectorised";
    info_.available = true;
    info_.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
    info_.detail["isa"] = ci.isa;
    info_.detail["heterogeneous_cores"] = ci.heterogeneous ? "yes" : (ci.metricReadable ? "no" : "unknown");
    info_.detail["topology_readable"] = ci.metricReadable ? "yes" : "no (sysfs hidden)";
    info_.detail["big_cores"] = std::to_string(ci.bigCores.size());
    info_.detail["policy"] = big_ ? "big cores only" : "all cores, dynamic scheduling";
  }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions&, std::string*) override {
    ThreadPool& pool = sharedPool(big_, reserve_);
    info_.detail["pinned_workers"] = std::to_string(pool.pinnedOk()) + "/" + std::to_string(pool.size());
    return std::unique_ptr<Execution>(new CpuExecution(p, pool));
  }

 private:
  bool big_;
  unsigned reserve_;
  BackendInfo info_;
};

}  // namespace

std::unique_ptr<Backend> makeCpuBackend(const DiscoverOptions& opt) {
  return std::unique_ptr<Backend>(new CpuBackend(opt));
}

}  // namespace hpc
