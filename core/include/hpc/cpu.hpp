// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hpc {

struct CpuInfo {
  unsigned logicalCores = 1;
  std::vector<int> cores;         // CPU ids this process may run on
  std::vector<double> capacity;   // per-core metric (cpu_capacity, else max kHz); 0 if unknown
  std::vector<int> bigCores;      // cores at >= 70% of the best metric (== cores if homogeneous/unknown)
  bool heterogeneous = false;     // more than one distinct performance class detected
  bool metricReadable = false;    // false => sysfs hidden; topology is a guess
  std::string arch;               // e.g. "aarch64", "x86_64"
  std::string isa;                // detected ISA extensions (informational)
};

CpuInfo probeCpu();

// Pin the calling thread. Returns false if the OS refused (reported, never fatal).
bool pinCurrentThread(int core);

// Persistent worker pool with dynamic chunk scheduling (cores that finish early take
// more chunks, so slow efficiency cores do not stall a parallel region).
class ThreadPool {
 public:
  ThreadPool(unsigned workers, const std::vector<int>& pinTo);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return nWorkers_; }
  unsigned pinnedOk() const { return pinnedOk_; }
  // fn(begin, end) is called for chunks of at most `grain` items covering [0,total).
  void parallelFor(uint32_t total, uint32_t grain, const std::function<void(uint32_t, uint32_t)>& fn);

 private:
  struct Impl;
  Impl* impl_;
  unsigned nWorkers_;
  unsigned pinnedOk_ = 0;
};

}  // namespace hpc
