// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdio>
#include <memory>
#include <string>

#include "hpc/backend.hpp"

namespace hpc {

// Placeholder for a backend that is not usable here; keeps the reason visible in reports.
class StubBackend final : public Backend {
 public:
  explicit StubBackend(BackendInfo info) : info_(std::move(info)) { info_.available = false; }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem&, const PrepareOptions&, std::string* err) override {
    if (err) *err = info_.note;
    return nullptr;
  }

 private:
  BackendInfo info_;
};

inline std::string vendorName(uint32_t id) {
  switch (id) {
    case 0x1002: return "AMD";
    case 0x10DE: return "NVIDIA";
    case 0x8086: return "Intel";
    case 0x13B5: return "Arm (Mali)";
    case 0x5143: return "Qualcomm (Adreno)";
    case 0x1010: return "Imagination (PowerVR)";
    case 0x1AE0: return "Google (SwiftShader)";
    case 0x144D: return "Samsung";
    case 0x14E4: return "Broadcom";
    case 0x106B: return "Apple";
    case 0x1414: return "Microsoft";
    case 0x19E5: return "Huawei";
    case 0x1D17: return "Zhaoxin";
    // Khronos-registered vendor ids (VkVendorId) for vendors without a PCI id
    case 0x10001: return "Vivante";
    case 0x10002: return "VeriSilicon";
    case 0x10003: return "Kazan";
    case 0x10004: return "Codeplay";
    case 0x10005: return "Mesa";
    case 0x10006: return "PoCL";
    case 0x10007: return "Mobileye";
    default: {
      char b[24];
      snprintf(b, sizeof b, "0x%04X", id);
      return b;
    }
  }
}

inline std::string hexStr(uint32_t v) {
  char b[16];
  snprintf(b, sizeof b, "0x%08X", v);
  return b;
}

}  // namespace hpc
