// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "hpc/log.hpp"

namespace hpc {

// Minimal dlopen/LoadLibrary wrapper. Vendor runtimes (OpenCL, NNAPI, Vulkan) are
// never linked at build time, so the app starts and degrades gracefully when one
// is missing.
//
// IMPORTANT: Loaded vendor libraries are intentionally NEVER unloaded (no dlclose /
// FreeLibrary during the process lifetime). Android GPU/OpenCL/NNAPI vendor drivers
// frequently spawn background threads, install pthread TLS destructors, or share
// internal state with the UI RenderThread; calling dlclose() after discovery unmaps
// code pages while driver threads are still active and causes SIGSEGV.
class DynLib {
 public:
  DynLib() = default;
  DynLib(const DynLib&) = delete;
  DynLib& operator=(const DynLib&) = delete;
  ~DynLib() { close(); }

  // Try each candidate name/path in order. Returns true on the first that loads.
  bool open(const std::vector<std::string>& candidates, std::string* tried = nullptr) {
    close();
    for (const auto& c : candidates) {
      if (c.empty()) continue;
      HPC_LOGV("[DL] dlopen attempt: %s", c.c_str());
#if defined(_WIN32)
      h_ = reinterpret_cast<void*>(LoadLibraryA(c.c_str()));
      if (!h_) {
        HPC_LOGV("[DL] LoadLibraryA failed for %s (err=%lu)", c.c_str(), static_cast<unsigned long>(GetLastError()));
      }
#else
      dlerror();  // clear any prior error
      h_ = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!h_) {
        const char* err = dlerror();
        HPC_LOGV("[DL] dlopen failed for %s: %s", c.c_str(), sanitizeAscii(err ? err : "unknown").c_str());
      }
#endif
      if (tried) *tried += (tried->empty() ? "" : ", ") + c;
      if (h_) {
        path_ = sanitizeAscii(c);
        HPC_LOGI("[DL] dlopen succeeded: %s (handle=%p)", path_.c_str(), h_);
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T sym(const char* name) const {
    if (!h_ || !name) return nullptr;
#if defined(_WIN32)
    void* p = reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h_), name));
#else
    dlerror();
    void* p = dlsym(h_, name);
#endif
    if (!p) {
      HPC_LOGV("[DL] dlsym missed '%s' in %s", name, path_.c_str());
    } else {
      HPC_LOGV("[DL] dlsym resolved '%s' in %s -> %p", name, path_.c_str(), p);
    }
    return reinterpret_cast<T>(p);
  }

  bool ok() const { return h_ != nullptr; }
  const std::string& path() const { return path_; }

  void close() {
    // Deliberately leak the native library handle for the lifetime of the process.
    // Never call dlclose() / FreeLibrary() on vendor driver libraries.
    h_ = nullptr;
  }

 private:
  void* h_ = nullptr;
  std::string path_;
};

}  // namespace hpc
