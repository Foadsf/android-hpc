// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hpc/types.hpp"

namespace hpc {

struct BackendInfo {
  std::string id;      // stable key, e.g. "cpu", "vulkan:0", "opencl:1", "gles", "nnapi:2"
  std::string api;     // human-readable API name
  std::string device;  // device / runtime name
  std::string vendor;
  std::string driver;
  bool available = false;
  std::string note;                           // why unavailable, or caveats
  std::map<std::string, std::string> detail;  // free-form capability probes
  std::vector<Kernel> kernels;                // kernels this backend can run
  // false when the backend is an API-compatible stand-in rather than the real
  // technology (see sycl_shim). Reported in every result so nothing is mislabelled.
  bool genuine = true;
};

// One prepared (compiled + uploaded) execution of a single Problem on a backend.
class Execution {
 public:
  virtual ~Execution() = default;
  // Run once and block until complete. deviceMs receives GPU-side time when the
  // API can measure it, otherwise a negative value.
  virtual bool run(double* deviceMs, std::string* err) = 0;
  // Copy the output buffer back to the host.
  virtual bool read(std::vector<float>& out, std::string* err) = 0;
  // True only if inputs were genuinely shared with the device without a copy.
  virtual bool zeroCopy() const { return false; }
};

struct PrepareOptions {
  bool zeroCopy = false;  // request AHardwareBuffer sharing where supported
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const BackendInfo& info() const = 0;
  // Compile pipelines and upload inputs. Not part of any timed region.
  virtual std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions& opt,
                                             std::string* err) = 0;
  bool supports(Kernel k) const {
    for (Kernel x : info().kernels)
      if (x == k) return true;
    return false;
  }
};

struct DiscoverOptions {
  bool cpuBigCoresOnly = false;  // pin CPU workers to the fastest cluster(s) only
  unsigned cpuReserveCores = 0;  // cores left free (the heterogeneous split reserves one for the GPU driver thread)
};

// Probe the machine and return every backend, including unavailable ones (their
// BackendInfo::note explains why).
std::vector<std::unique_ptr<Backend>> discoverBackends(const DiscoverOptions& opt);

// Individual factories (implemented per backend; some compile to stubs on a platform).
std::unique_ptr<Backend> makeCpuBackend(const DiscoverOptions& opt);
std::unique_ptr<Backend> makeSyclShimBackend();
void discoverVulkan(std::vector<std::unique_ptr<Backend>>& out);
void discoverOpenCL(std::vector<std::unique_ptr<Backend>>& out);
void discoverGles(std::vector<std::unique_ptr<Backend>>& out);
void discoverNnapi(std::vector<std::unique_ptr<Backend>>& out);

}  // namespace hpc
