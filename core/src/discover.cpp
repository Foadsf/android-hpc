// SPDX-License-Identifier: Apache-2.0
#include "hpc/backend.hpp"

#include <exception>

#include "backend_util.hpp"
#include "hpc/log.hpp"

namespace hpc {

namespace {

void addStub(std::vector<std::unique_ptr<Backend>>& out, const char* id, const char* api, const std::string& note) {
  BackendInfo i;
  i.id = id ? id : "unknown";
  i.api = api ? api : "unknown";
  i.available = false;
  i.note = sanitizeAscii(note);
  out.emplace_back(new StubBackend(std::move(i)));
}

template <typename Fn>
void runFamilyDiscovery(std::vector<std::unique_ptr<Backend>>& out, const char* family, const char* apiName, Fn&& fn) {
  const std::string beginCrumb = std::string("discover:") + family + ":begin";
  const std::string endCrumb = std::string("discover:") + family + ":end";
  setBreadcrumb(beginCrumb);
  if (isBackendDisabled(family)) {
    HPC_LOGW("[DISCOVER] skipping family '%s' (quarantined / disabled after a crash)", family);
    addStub(out, family, apiName, "disabled after a crash");
    setBreadcrumb(endCrumb);
    return;
  }
  const size_t before = out.size();
  try {
    HPC_LOGI("[DISCOVER] probing family '%s'...", family);
    fn(out);
    HPC_LOGI("[DISCOVER] family '%s' added %zu backend entry(s)", family, out.size() - before);
  } catch (const std::exception& e) {
    const std::string why = std::string("exception during discovery: ") + sanitizeAscii(e.what());
    HPC_LOGE("[DISCOVER] family '%s' threw std::exception: %s", family, why.c_str());
    if (out.size() == before) addStub(out, family, apiName, why);
  } catch (...) {
    const std::string why = "unknown non-std exception during discovery";
    HPC_LOGE("[DISCOVER] family '%s' threw unknown exception", family);
    if (out.size() == before) addStub(out, family, apiName, why);
  }
  setBreadcrumb(endCrumb);
}

}  // namespace

std::vector<std::unique_ptr<Backend>> discoverBackends(const DiscoverOptions& opt) {
  HPC_LOGI("[DISCOVER] starting discoverBackends (bigCoresOnly=%d, reserveCores=%u)",
           opt.cpuBigCoresOnly ? 1 : 0, opt.cpuReserveCores);
  std::vector<std::unique_ptr<Backend>> v;

  try {
    v.push_back(makeCpuBackend(opt));
  } catch (const std::exception& e) {
    HPC_LOGE("[DISCOVER] CPU backend init failed: %s", sanitizeAscii(e.what()).c_str());
    addStub(v, "cpu", "CPU (C++17 threads)", std::string("exception: ") + sanitizeAscii(e.what()));
  } catch (...) {
    HPC_LOGE("[DISCOVER] CPU backend init threw unknown exception");
    addStub(v, "cpu", "CPU (C++17 threads)", "unknown exception during CPU init");
  }

  runFamilyDiscovery(v, "vulkan", "Vulkan compute (SPIR-V)", discoverVulkan);
  runFamilyDiscovery(v, "opencl", "OpenCL", discoverOpenCL);
  runFamilyDiscovery(v, "gles", "OpenGL ES 3.1 compute", discoverGles);
  runFamilyDiscovery(v, "nnapi", "NNAPI", discoverNnapi);

  try {
    v.push_back(makeSyclShimBackend());
  } catch (const std::exception& e) {
    HPC_LOGE("[DISCOVER] SYCL-shim backend init failed: %s", sanitizeAscii(e.what()).c_str());
    addStub(v, "sycl-shim", "SYCL-style (NOT real SYCL)", std::string("exception: ") + sanitizeAscii(e.what()));
  } catch (...) {
    HPC_LOGE("[DISCOVER] SYCL-shim backend init threw unknown exception");
    addStub(v, "sycl-shim", "SYCL-style (NOT real SYCL)", "unknown exception");
  }

  HPC_LOGI("[DISCOVER] discoverBackends finished: %zu total backend entries", v.size());
  return v;
}

}  // namespace hpc
