// SPDX-License-Identifier: Apache-2.0
// Used when the build had no glslc or no Vulkan headers.
#include "backend_util.hpp"

namespace hpc {
void discoverVulkan(std::vector<std::unique_ptr<Backend>>& out) {
  BackendInfo i;
  i.id = "vulkan";
  i.api = "Vulkan compute (SPIR-V)";
  i.note = "not compiled into this build (glslc or Vulkan headers were missing at configure time)";
  out.emplace_back(new StubBackend(i));
}
}  // namespace hpc
