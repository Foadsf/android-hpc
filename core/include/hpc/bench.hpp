// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "hpc/backend.hpp"
#include "hpc/cpu.hpp"

namespace hpc {

struct BenchConfig {
  std::string preset = "medium";  // small | medium | large (see presetSize)
  uint32_t sizeOverride[5] = {0, 0, 0, 0, 0};  // indexed by Kernel; 0 = use preset
  int warmup = 3;
  int iterations = 10;
  std::vector<std::string> backends;  // id or id-prefix filters ("cpu", "vulkan", "opencl:1"); empty = all
  std::vector<Kernel> kernels;        // empty = all
  bool zeroCopy = false;              // request AHardwareBuffer sharing (Vulkan, Android)
  bool hetero = true;                 // run the CPU+GPU split experiment
  bool cpuBigOnly = false;            // pin CPU workers to the fastest cluster(s)
};

uint32_t presetSize(const std::string& preset, Kernel k);

struct KernelResult {
  std::string backendId, api, device;
  bool genuine = true;
  std::string kernel, size;
  double flops = 0, bytes = 0;
  bool ran = false;        // false: prepare/run failed (see error)
  std::string error;
  double prepareMs = 0;
  Stats wallMs, devMs;     // devMs.count == 0 when the API gave no device timing
  double gflops = 0, gbps = 0;  // from device median if available, else wall median
  std::string timeBasis;   // "device" or "wall"
  bool verified = false;
  double maxAbsErr = 0, maxRelErr = 0;
  bool zeroCopy = false;
};

struct HeteroResult {
  bool ran = false;
  std::string error, gpuId, kernel = "sgemm", size;
  uint32_t cpuRows = 0, gpuRows = 0;
  Stats wallMs;
  double cpuAloneMs = 0, gpuAloneMs = 0, speedupVsBest = 0;
  bool verified = false;
  double maxAbsErr = 0;
};

struct Report {
  std::string version, platform, preset;
  CpuInfo cpu;
  int warmup = 0, iterations = 0;
  std::vector<BackendInfo> backends;
  std::vector<KernelResult> results;
  HeteroResult hetero;
  bool allVerified = true;  // every attempted run produced a verified result
  bool cancelled = false;
};

using ProgressFn = std::function<void(const std::string&)>;

Report discoverReport(const DiscoverOptions& opt);
Report runBenchmarks(const BenchConfig& cfg, const ProgressFn& progress, std::atomic<bool>* cancel);

std::string toJson(const Report& r);
std::string toText(const Report& r);

const char* versionString();

}  // namespace hpc
