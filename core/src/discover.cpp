// SPDX-License-Identifier: Apache-2.0
#include "hpc/backend.hpp"

namespace hpc {

std::vector<std::unique_ptr<Backend>> discoverBackends(const DiscoverOptions& opt) {
  std::vector<std::unique_ptr<Backend>> v;
  v.push_back(makeCpuBackend(opt));
  discoverVulkan(v);
  discoverOpenCL(v);
  discoverGles(v);
  discoverNnapi(v);
  v.push_back(makeSyclShimBackend());
  return v;
}

}  // namespace hpc
