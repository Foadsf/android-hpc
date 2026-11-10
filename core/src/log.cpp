// SPDX-License-Identifier: Apache-2.0
#include "hpc/log.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__ANDROID__) || defined(__GLIBC__)
#include <unwind.h>
#endif
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/syscall.h>
#endif
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace hpc {
namespace {

std::mutex gLogMu;
std::string gLogPath;
std::string gBreadcrumbPath;
int gLogFd = -1;
int gBreadcrumbFd = -1;
std::atomic<bool> gVerbose{true};
char gLastBreadcrumb[256] = "init";

std::mutex gQuarantineMu;
std::set<std::string> gDisabledFamilies;

const auto gStartClock = std::chrono::steady_clock::now();

uint64_t elapsedMs() {
  return uint64_t(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gStartClock).count());
}

uint64_t currentTid() {
#if defined(__ANDROID__) || defined(__linux__)
  return uint64_t(syscall(SYS_gettid));
#elif defined(_WIN32)
  return uint64_t(GetCurrentThreadId());
#else
  return uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFFFFu);
#endif
}

void closeFd(int& fd) {
  if (fd >= 0) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
    fd = -1;
  }
}

void writeAllFd(int fd, const char* buf, size_t len, bool doSync) {
  if (fd < 0 || !buf || len == 0) return;
  size_t off = 0;
  while (off < len) {
#if defined(_WIN32)
    int w = _write(fd, buf + off, unsigned(len - off));
#else
    ssize_t w = write(fd, buf + off, len - off);
    if (w < 0 && errno == EINTR) continue;
#endif
    if (w <= 0) break;
    off += size_t(w);
  }
  if (doSync) {
#if defined(_WIN32)
    _commit(fd);
#else
    fsync(fd);
#endif
  }
}

const char* levelTag(LogLevel l) {
  switch (l) {
    case LogLevel::Verbose: return "V";
    case LogLevel::Info: return "I";
    case LogLevel::Warn: return "W";
    case LogLevel::Error: return "E";
  }
  return "I";
}

#if !defined(_WIN32)
// ---- Async-signal-safe crash handler (POSIX / Android) --------------------------------------
// Limits:
// 1. Only catches synchronous hardware/abort signals in-process (SIGSEGV, SIGBUS, SIGILL,
//    SIGFPE, SIGABRT). Cannot catch SIGKILL, out-of-memory kills by Android lmkd, or full
//    system/GPU kernel panics. However, the fsynced breadcrumb file on disk survives all of those.
// 2. _Unwind_Backtrace and dladdr are best-effort inside a signal handler: if the stack pointer
//    or dynamic linker lock is corrupted, unwinding may stop early after writing the header and
//    last breadcrumb.
alignas(16) uint8_t gAltStack[64 * 1024];
std::atomic<bool> gHandlersInstalled{false};
constexpr int kCrashSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT};
struct sigaction gPrevActions[sizeof(kCrashSignals) / sizeof(kCrashSignals[0])];

// Put back whatever handler was installed before ours (on Android: debuggerd's, which writes
// the tombstone that `adb logcat -b crash` shows). Async-signal-safe: sigaction only.
void restorePreviousHandler(int sig) {
  for (size_t i = 0; i < sizeof(kCrashSignals) / sizeof(kCrashSignals[0]); ++i)
    if (kCrashSignals[i] == sig) {
      sigaction(sig, &gPrevActions[i], nullptr);
      return;
    }
  signal(sig, SIG_DFL);
}

struct CrashBuf {
  char data[4096];
  size_t len = 0;

  void putc(char c) {
    if (len + 1 < sizeof(data)) data[len++] = c;
  }
  void puts(const char* s) {
    if (!s) return;
    while (*s && len + 1 < sizeof(data)) data[len++] = *s++;
  }
  void putDec(uint64_t v) {
    char tmp[24];
    int i = 0;
    if (v == 0) {
      putc('0');
      return;
    }
    while (v > 0 && i < 23) {
      tmp[i++] = char('0' + (v % 10));
      v /= 10;
    }
    while (i > 0) putc(tmp[--i]);
  }
  void putHex( uintptr_t v) {
    puts("0x");
    char tmp[20];
    int i = 0;
    if (v == 0) {
      putc('0');
      return;
    }
    while (v > 0 && i < 18) {
      unsigned d = unsigned(v & 0xFu);
      tmp[i++] = char(d < 10 ? ('0' + d) : ('a' + (d - 10)));
      v >>= 4;
    }
    while (i > 0) putc(tmp[--i]);
  }
};

const char* sigName(int sig) {
  switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    case SIGABRT: return "SIGABRT";
    default: return "SIGNAL";
  }
}

#if defined(__ANDROID__) || defined(__GLIBC__)
struct UnwindState {
  CrashBuf* cb;
  int depth;
};

_Unwind_Reason_Code unwindStep(struct _Unwind_Context* ctx, void* arg) {
  UnwindState* st = static_cast<UnwindState*>(arg);
  if (st->depth >= 24) return _URC_END_OF_STACK;
  uintptr_t pc = _Unwind_GetIP(ctx);
  if (pc) {
    st->cb->puts("  #");
    st->cb->putDec(uint64_t(st->depth));
    st->cb->puts(" pc ");
    st->cb->putHex(pc);
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fname) {
      st->cb->puts(" ");
      st->cb->puts(info.dli_fname);
      if (info.dli_fbase) {
        uintptr_t base = reinterpret_cast<uintptr_t>(info.dli_fbase);
        if (pc >= base) {
          st->cb->puts(" (+");
          st->cb->putHex(pc - base);
          st->cb->puts(")");
        }
      }
      if (info.dli_sname) {
        st->cb->puts(" [");
        st->cb->puts(info.dli_sname);
        st->cb->puts("]");
      }
    }
    st->cb->putc('\n');
    ++st->depth;
  }
  return _URC_NO_REASON;
}
#endif

void crashSignalHandler(int sig, siginfo_t* info, void* /*ucontext*/) {
  CrashBuf cb;
  cb.puts("\n=== NATIVE CRASH ===\n");
  cb.puts("signal: ");
  cb.putDec(uint64_t(sig));
  cb.puts(" (");
  cb.puts(sigName(sig));
  cb.puts(") code=");
  cb.putDec(uint64_t(info ? info->si_code : 0));
  cb.puts(" fault_addr=");
  cb.putHex(info ? reinterpret_cast<uintptr_t>(info->si_addr) : 0);
  cb.puts(" tid=");
  cb.putDec(currentTid());
  cb.puts("\nlast_breadcrumb: ");
  cb.puts(gLastBreadcrumb);
  cb.putc('\n');

#if defined(__ANDROID__) || defined(__GLIBC__)
  cb.puts("backtrace:\n");
  UnwindState st{&cb, 0};
  _Unwind_Backtrace(unwindStep, &st);
#endif
  cb.puts("=== END NATIVE CRASH ===\n");

  const int fd = gLogFd;
  if (fd >= 0) {
    writeAllFd(fd, cb.data, cb.len, true);
  }
  writeAllFd(STDERR_FILENO, cb.data, cb.len, false);

  // Chain to the previous handler instead of SIG_DFL, so Android's debuggerd still produces a
  // tombstone. A fault raised by an instruction (si_code > 0) re-executes on return and hits the
  // restored handler with the original context; a signal sent by abort()/kill is re-raised.
  restorePreviousHandler(sig);
  if (!info || info->si_code <= 0 || sig == SIGABRT) raise(sig);
}
#endif  // !defined(_WIN32)

}  // namespace

void initLog(const std::string& logPath, const std::string& breadcrumbPath, bool verbose) {
  std::lock_guard<std::mutex> lk(gLogMu);
  gVerbose.store(verbose);
  if (logPath != gLogPath) {
    closeFd(gLogFd);
    gLogPath = logPath;
    if (!gLogPath.empty()) {
#if defined(_WIN32)
      gLogFd = _open(gLogPath.c_str(), _O_CREAT | _O_WRONLY | _O_APPEND | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
      gLogFd = open(gLogPath.c_str(), O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0644);
#endif
    }
  }
  if (breadcrumbPath != gBreadcrumbPath) {
    closeFd(gBreadcrumbFd);
    gBreadcrumbPath = breadcrumbPath;
    if (!gBreadcrumbPath.empty()) {
#if defined(_WIN32)
      gBreadcrumbFd =
          _open(gBreadcrumbPath.c_str(), _O_CREAT | _O_RDWR | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
      gBreadcrumbFd = open(gBreadcrumbPath.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0644);
#endif
    }
  }
}

void setLogVerbose(bool verbose) { gVerbose.store(verbose); }

bool isLogVerbose() { return gVerbose.load(); }

std::string currentLogPath() {
  std::lock_guard<std::mutex> lk(gLogMu);
  return gLogPath;
}

std::string currentBreadcrumbPath() {
  std::lock_guard<std::mutex> lk(gLogMu);
  return gBreadcrumbPath;
}

void setBreadcrumb(const char* crumb) {
  const std::string safe = sanitizeAscii(crumb ? crumb : "", 240);
  {
    std::lock_guard<std::mutex> lk(gLogMu);
    std::snprintf(gLastBreadcrumb, sizeof(gLastBreadcrumb), "%s", safe.c_str());
    if (!gBreadcrumbPath.empty()) {
      if (gBreadcrumbFd >= 0) {
#if defined(_WIN32)
        _lseek(gBreadcrumbFd, 0, SEEK_SET);
        _chsize(gBreadcrumbFd, 0);
#else
        lseek(gBreadcrumbFd, 0, SEEK_SET);
        if (ftruncate(gBreadcrumbFd, 0) != 0) { /* best effort */ }
#endif
        std::string line = safe + "\n";
        writeAllFd(gBreadcrumbFd, line.data(), line.size(), true);
      }
    }
  }
  logPrint(LogLevel::Info, "BREADCRUMB %s", safe.c_str());
}

std::string lastBreadcrumb() {
  std::lock_guard<std::mutex> lk(gLogMu);
  return std::string(gLastBreadcrumb);
}

void installCrashHandlers() {
#if !defined(_WIN32)
  bool expected = false;
  if (!gHandlersInstalled.compare_exchange_strong(expected, true)) return;

  stack_t ss{};
  ss.ss_sp = gAltStack;
  ss.ss_size = sizeof(gAltStack);
  ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);

  struct sigaction sa{};
  sa.sa_sigaction = crashSignalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;  // no SA_RESETHAND: the handler restores the previous one

  for (size_t i = 0; i < sizeof(kCrashSignals) / sizeof(kCrashSignals[0]); ++i) {
    if (sigaction(kCrashSignals[i], &sa, &gPrevActions[i]) != 0) {
      gPrevActions[i] = {};
      gPrevActions[i].sa_handler = SIG_DFL;
    }
  }
#endif
}

void logPrint(LogLevel level, const char* fmt, ...) {
  const bool verb = gVerbose.load();
  if (level == LogLevel::Verbose && !verb) return;
  if (!fmt) return;

  char msg[1536];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);

  const std::string clean = sanitizeAscii(msg, sizeof(msg) - 1);
  char line[1700];
  int n = std::snprintf(line, sizeof(line), "[%8llu ms][tid %llu][%s] %s\n",
                        static_cast<unsigned long long>(elapsedMs()),
                        static_cast<unsigned long long>(currentTid()), levelTag(level), clean.c_str());
  if (n < 0) return;
  size_t len = std::min(size_t(n), sizeof(line) - 1);

  {
    std::lock_guard<std::mutex> lk(gLogMu);
    if (gLogFd >= 0) {
      const bool syncNow = verb || level >= LogLevel::Warn || std::strncmp(clean.c_str(), "BREADCRUMB ", 11) == 0;
      writeAllFd(gLogFd, line, len, syncNow);
    }
#if !defined(__ANDROID__)
    if (verb || level >= LogLevel::Warn) {
      std::fwrite(line, 1, len, stderr);
      std::fflush(stderr);
    }
#endif
  }

#if defined(__ANDROID__)
  int prio = ANDROID_LOG_INFO;
  switch (level) {
    case LogLevel::Verbose: prio = ANDROID_LOG_VERBOSE; break;
    case LogLevel::Info: prio = ANDROID_LOG_INFO; break;
    case LogLevel::Warn: prio = ANDROID_LOG_WARN; break;
    case LogLevel::Error: prio = ANDROID_LOG_ERROR; break;
  }
  __android_log_print(prio, "android-hpc", "%s", clean.c_str());
#endif
}

std::string sanitizeAscii(const char* s, size_t maxLen) {
  if (!s) return "";
  std::string out;
  for (size_t i = 0; s[i] != '\0' && i < maxLen; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '\n' || c == '\t') {
      out.push_back(char(c));
    } else if (c == '\r') {
      // Drop carriage return to normalize line endings.
    } else if (c >= 0x20 && c <= 0x7E) {
      out.push_back(char(c));
    } else {
      out.push_back('?');
    }
  }
  return out;
}

std::string sanitizeAscii(const std::string& s, size_t maxLen) {
  std::string out;
  const size_t limit = std::min(s.size(), maxLen);
  out.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '\n' || c == '\t') {
      out.push_back(char(c));
    } else if (c == '\r') {
      // Drop CR
    } else if (c >= 0x20 && c <= 0x7E) {
      out.push_back(char(c));
    } else {
      out.push_back('?');
    }
  }
  return out;
}

void setDisabledBackends(const std::string& csv) {
  std::set<std::string> next;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed;
    for (char c : item) {
      if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
        trimmed.push_back(char(c >= 'A' && c <= 'Z' ? (c - 'A' + 'a') : c));
      }
    }
    if (!trimmed.empty()) next.insert(trimmed);
  }
  {
    std::lock_guard<std::mutex> lk(gQuarantineMu);
    gDisabledFamilies = std::move(next);
  }
  HPC_LOGI("Disabled backend families set to: '%s'", csv.c_str());
}

bool isBackendDisabled(const char* family) {
  if (!family || !*family) return false;
  std::string key;
  for (const char* p = family; *p && *p != ':'; ++p) {
    char c = *p;
    key.push_back(char(c >= 'A' && c <= 'Z' ? (c - 'A' + 'a') : c));
  }
  std::lock_guard<std::mutex> lk(gQuarantineMu);
  return gDisabledFamilies.count(key) > 0;
}

}  // namespace hpc
