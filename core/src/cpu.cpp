// SPDX-License-Identifier: Apache-2.0
#include "hpc/cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <sys/auxv.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace hpc {

namespace {
bool readDouble(const std::string& path, double& v) {
  std::ifstream f(path);
  return f.is_open() && (f >> v);
}
}  // namespace

CpuInfo probeCpu() {
  CpuInfo ci;
#if defined(__aarch64__)
  ci.arch = "aarch64";
#elif defined(__arm__)
  ci.arch = "armv7";
#elif defined(__x86_64__) || defined(_M_X64)
  ci.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  ci.arch = "x86";
#else
  ci.arch = "unknown";
#endif

#if defined(__linux__) || defined(__ANDROID__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0)
    for (int i = 0; i < CPU_SETSIZE; ++i)
      if (CPU_ISSET(i, &set)) ci.cores.push_back(i);
#endif
  if (ci.cores.empty()) {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < n; ++i) ci.cores.push_back(int(i));
  }
  ci.logicalCores = unsigned(ci.cores.size());

  // Performance class per core: prefer cpu_capacity (kernel's own scale), else max frequency.
  ci.capacity.assign(ci.cores.size(), 0.0);
#if defined(__linux__) || defined(__ANDROID__)
  for (size_t i = 0; i < ci.cores.size(); ++i) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(ci.cores[i]);
    double v = 0;
    if (readDouble(base + "/cpu_capacity", v) && v > 0) ci.capacity[i] = v;
    else if (readDouble(base + "/cpufreq/cpuinfo_max_freq", v) && v > 0) ci.capacity[i] = v;
  }
#endif
  double best = 0;
  for (double c : ci.capacity) best = std::max(best, c);
  ci.metricReadable = best > 0;
  if (ci.metricReadable) {
    double lo = best;
    for (double c : ci.capacity) lo = std::min(lo, c > 0 ? c : best);
    for (size_t i = 0; i < ci.cores.size(); ++i)
      if (ci.capacity[i] >= 0.7 * best) ci.bigCores.push_back(ci.cores[i]);
    ci.heterogeneous = lo < 0.9 * best;
  }
  if (ci.bigCores.empty()) ci.bigCores = ci.cores;

  // ISA extensions: informational only. The kernels are portable C++ built for the
  // baseline ISA; they do not dispatch on these bits.
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
  const unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
  ci.isa = "NEON";
  if (h1 & (1UL << 10)) ci.isa += " FP16";
  if (h1 & (1UL << 20)) ci.isa += " DOTPROD";
  if (h1 & (1UL << 22)) ci.isa += " SVE";
  if (h2 & (1UL << 1)) ci.isa += " SVE2";
  if (h2 & (1UL << 13)) ci.isa += " I8MM";
  if (h2 & (1UL << 14)) ci.isa += " BF16";
#elif defined(__x86_64__) || defined(_M_X64)
  ci.isa = "x86-64 (SSE2 baseline)";
#endif
  return ci;
}

bool pinCurrentThread(int core) {
#if defined(__linux__) || defined(__ANDROID__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;  // pid 0 == calling thread on Linux
#elif defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << core) != 0;
#else
  (void)core;
  return false;
#endif
}

struct ThreadPool::Impl {
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable cvWork, cvDone;
  std::mutex submitMu;  // one parallelFor at a time
  uint64_t generation = 0;
  bool stop = false;
  unsigned done = 0;
  const std::function<void(uint32_t, uint32_t)>* fn = nullptr;
  uint32_t total = 0, grain = 1;
  std::atomic<uint32_t> next{0};
  std::atomic<unsigned> pinOk{0};
};

ThreadPool::ThreadPool(unsigned workers, const std::vector<int>& pinTo)
    : impl_(new Impl), nWorkers_(std::max(1u, workers)) {
  for (unsigned w = 0; w < nWorkers_; ++w) {
    const int core = pinTo.empty() ? -1 : pinTo[w % pinTo.size()];
    impl_->threads.emplace_back([this, core] {
      if (core >= 0 && pinCurrentThread(core)) impl_->pinOk++;
      uint64_t seen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(impl_->mu);
        impl_->cvWork.wait(lk, [&] { return impl_->stop || impl_->generation != seen; });
        if (impl_->stop) return;
        seen = impl_->generation;
        const auto* fn = impl_->fn;
        const uint32_t total = impl_->total, grain = impl_->grain;
        lk.unlock();
        for (;;) {
          const uint32_t b = impl_->next.fetch_add(grain);
          if (b >= total) break;
          (*fn)(b, std::min(total, b + grain));
        }
        lk.lock();
        if (++impl_->done == nWorkers_) impl_->cvDone.notify_one();
      }
    });
  }
  // Give workers a moment to record pin results (informational only).
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  pinnedOk_ = impl_->pinOk.load();
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->stop = true;
  }
  impl_->cvWork.notify_all();
  for (auto& t : impl_->threads) t.join();
  delete impl_;
}

void ThreadPool::parallelFor(uint32_t total, uint32_t grain,
                             const std::function<void(uint32_t, uint32_t)>& fn) {
  if (total == 0) return;
  std::lock_guard<std::mutex> sub(impl_->submitMu);
  std::unique_lock<std::mutex> lk(impl_->mu);
  impl_->fn = &fn;
  impl_->total = total;
  impl_->grain = std::max(1u, grain);
  impl_->next.store(0);
  impl_->done = 0;
  ++impl_->generation;
  lk.unlock();
  impl_->cvWork.notify_all();
  lk.lock();
  impl_->cvDone.wait(lk, [&] { return impl_->done == nWorkers_; });
}

}  // namespace hpc
