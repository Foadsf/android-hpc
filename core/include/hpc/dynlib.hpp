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

namespace hpc {

// Minimal dlopen/LoadLibrary wrapper. Vendor runtimes (OpenCL, NNAPI, Vulkan) are
// never linked at build time, so the app starts and degrades gracefully when one
// is missing.
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
#if defined(_WIN32)
      h_ = reinterpret_cast<void*>(LoadLibraryA(c.c_str()));
#else
      h_ = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
      if (tried) *tried += (tried->empty() ? "" : ", ") + c;
      if (h_) {
        path_ = c;
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T sym(const char* name) const {
    if (!h_) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<T>(GetProcAddress(reinterpret_cast<HMODULE>(h_), name));
#else
    return reinterpret_cast<T>(dlsym(h_, name));
#endif
  }

  bool ok() const { return h_ != nullptr; }
  const std::string& path() const { return path_; }

  void close() {
    if (!h_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(h_));
#else
    dlclose(h_);
#endif
    h_ = nullptr;
  }

 private:
  void* h_ = nullptr;
  std::string path_;
};

}  // namespace hpc
