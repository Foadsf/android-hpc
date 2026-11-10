// SPDX-License-Identifier: Apache-2.0
// hpc_cli: run the benchmark suite from a shell (host PC or `adb shell`).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "hpc/bench.hpp"
#include "hpc/log.hpp"

namespace {
void usage() {
  std::puts(
      "hpc_cli [options]\n"
      "  --preset small|medium|large   problem sizes (default medium)\n"
      "  --kernel a,b,...              sgemm,conv3x3,reduce,triad,dense (default all)\n"
      "  --backend a,b,...             ids or prefixes: cpu,vulkan,opencl,gles,nnapi,sycl-shim,vulkan:1 (default all)\n"
      "  --size kernel=N               override one problem size (repeatable)\n"
      "  --warmup N  --iters N         warm-up and timed iterations (default 3 / 10)\n"
      "  --big-cores                   pin CPU workers to the fastest cluster(s) only\n"
      "  --zero-copy                   request AHardwareBuffer sharing (Vulkan on Android; experimental)\n"
      "  --no-hetero                   skip the CPU+GPU split experiment\n"
      "  --verbose                     enable verbose debug logging\n"
      "  --log-file FILE               write persistent debug log to FILE\n"
      "  --json FILE                   also write the full report as JSON\n"
      "  --list                        discover and print backends, then exit\n"
      "Exit status: 0 all attempted runs verified, 1 a run failed or mismatched the fp64 reference, 2 usage error.");
}
// Strict non-negative integer parse: rejects "", "abc", "12x", "-3" and values above `maxV`.
bool parseUInt(const std::string& s, unsigned long maxV, unsigned long& out) {
  if (s.empty() || s.size() > 12) return false;
  for (char c : s) if (c < '0' || c > '9') return false;
  out = std::strtoul(s.c_str(), nullptr, 10);
  return out <= maxV;
}
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  hpc::BenchConfig cfg;
  std::string jsonPath;
  std::string logPath;
  bool verbose = false;
  bool list = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
    if (a == "--help" || a == "-h") { usage(); return 0; }
    else if (a == "--preset") cfg.preset = next();
    else if (a == "--kernel") {
      for (auto& s : split(next())) { hpc::Kernel k; if (!hpc::parseKernel(s, k)) { std::fprintf(stderr, "unknown kernel: %s\n", s.c_str()); return 2; } cfg.kernels.push_back(k); }
    } else if (a == "--backend") cfg.backends = split(next());
    else if (a == "--size") {
      const std::string v = next();
      const auto eq = v.find('=');
      hpc::Kernel k;
      if (eq == std::string::npos || !hpc::parseKernel(v.substr(0, eq), k)) { std::fprintf(stderr, "bad --size %s (want kernel=N)\n", v.c_str()); return 2; }
      unsigned long n = 0;
      if (!parseUInt(v.substr(eq + 1), 1ul << 30, n) || n == 0) { std::fprintf(stderr, "bad --size %s (N must be 1..2^30)\n", v.c_str()); return 2; }
      cfg.sizeOverride[int(k)] = uint32_t(n);
    } else if (a == "--warmup" || a == "--iters") {
      const std::string v = next();
      unsigned long n = 0;
      if (!parseUInt(v, 100000, n) || (a == "--iters" && n == 0)) { std::fprintf(stderr, "bad %s %s\n", a.c_str(), v.c_str()); return 2; }
      (a == "--warmup" ? cfg.warmup : cfg.iterations) = int(n);
    }
    else if (a == "--big-cores") cfg.cpuBigOnly = true;
    else if (a == "--zero-copy") cfg.zeroCopy = true;
    else if (a == "--no-hetero") cfg.hetero = false;
    else if (a == "--verbose") verbose = true;
    else if (a == "--log-file") logPath = next();
    else if (a == "--json") jsonPath = next();
    else if (a == "--list") list = true;
    else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
  }
  if (cfg.preset != "small" && cfg.preset != "medium" && cfg.preset != "large") { std::fprintf(stderr, "bad preset\n"); return 2; }

  hpc::initLog(logPath, logPath.empty() ? "" : (logPath + ".breadcrumb"), verbose);
  hpc::installCrashHandlers();

  if (list) {
    hpc::DiscoverOptions d;
    d.cpuBigCoresOnly = cfg.cpuBigOnly;
    std::cout << hpc::toText(hpc::discoverReport(d));
    return 0;
  }
  const hpc::Report rep = hpc::runBenchmarks(cfg, [](const std::string& s) { std::fprintf(stderr, "%s\n", s.c_str()); }, nullptr);
  std::cout << hpc::toText(rep);
  if (!jsonPath.empty()) {
    std::ofstream f(jsonPath, std::ios::binary);
    f << hpc::toJson(rep);
    f.close();
    if (!f) { std::fprintf(stderr, "error: could not write %s\n", jsonPath.c_str()); return 1; }
    std::fprintf(stderr, "wrote %s\n", jsonPath.c_str());
  }
  bool anyRan = false;
  for (const auto& r : rep.results) anyRan |= r.ran;
  return (rep.allVerified && anyRan) ? 0 : 1;
}
