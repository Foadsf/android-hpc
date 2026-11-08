// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <string>

namespace hpc {

enum class LogLevel { Verbose = 0, Info = 1, Warn = 2, Error = 3 };

// Initialize file + breadcrumb logging. Safe to call multiple times; reopens files if paths change.
void initLog(const std::string& logPath, const std::string& breadcrumbPath, bool verbose);
void setLogVerbose(bool verbose);
bool isLogVerbose();
std::string currentLogPath();
std::string currentBreadcrumbPath();

// Record a durable breadcrumb (written and fsynced to both the breadcrumb file and the main log).
void setBreadcrumb(const char* crumb);
inline void setBreadcrumb(const std::string& crumb) { setBreadcrumb(crumb.c_str()); }
std::string lastBreadcrumb();

// Install async-signal-safe native crash handlers (SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT).
void installCrashHandlers();

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void logPrint(LogLevel level, const char* fmt, ...);

// Replace non-printable / non-ASCII bytes with '?' so strings from vendor drivers are always
// valid ASCII / Modified UTF-8 for JNI NewStringUTF and JSON/text output.
std::string sanitizeAscii(const char* s, size_t maxLen = 512);
std::string sanitizeAscii(const std::string& s, size_t maxLen = 512);

#define HPC_LOGV(...) ::hpc::logPrint(::hpc::LogLevel::Verbose, __VA_ARGS__)
#define HPC_LOGI(...) ::hpc::logPrint(::hpc::LogLevel::Info, __VA_ARGS__)
#define HPC_LOGW(...) ::hpc::logPrint(::hpc::LogLevel::Warn, __VA_ARGS__)
#define HPC_LOGE(...) ::hpc::logPrint(::hpc::LogLevel::Error, __VA_ARGS__)


// ---- crash quarantine (contract between the Java app and the native backends) ----
// Comma-separated backend FAMILIES to skip entirely: any of vulkan, opencl, gles, nnapi.
void setDisabledBackends(const std::string& csv);
bool isBackendDisabled(const char* family);

// Breadcrumb protocol. Native code calls setBreadcrumb() immediately before and after every
// risky vendor-driver step, using EXACTLY these forms:
//   "discover:<family>:begin"  /  "discover:<family>:end"
//   "prepare:<backendId>:<kernel>:begin" / "...:end"      (e.g. prepare:vulkan:1:sgemm:begin)
//   "run:<backendId>:<kernel>:begin" / "...:end"
// Java, after an unclean exit, reads the LAST breadcrumb; if it ends in ":begin" the family
// (discover) or the backendId's family (prepare/run) is quarantined.
}  // namespace hpc
