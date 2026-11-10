// SPDX-License-Identifier: Apache-2.0
#include "hpc/bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <thread>

#include <exception>

#include "hpc/log.hpp"
#include "hpc/reference.hpp"

namespace hpc {

#ifndef HPC_VERSION
#define HPC_VERSION "0.0.0-unversioned"
#endif
const char* versionString() { return HPC_VERSION; }

uint32_t presetSize(const std::string& preset, Kernel k) {
  // Sizes keep peak RAM (problem + fp64 reference + device copies) modest on phones.
  static const uint32_t small[] = {256, 1024, 2u << 20, 2u << 20, 256};    // sgemm conv reduce triad dense
  static const uint32_t medium[] = {512, 2048, 8u << 20, 8u << 20, 512};
  static const uint32_t large[] = {1024, 4096, 32u << 20, 16u << 20, 1024};
  const uint32_t* t = preset == "small" ? small : preset == "large" ? large : medium;
  return t[int(k)];
}

bool backendMatches(const std::string& id, const std::vector<std::string>& filters) {
  if (filters.empty()) return true;
  for (const auto& s : filters)
    if (id == s || id.rfind(s + ":", 0) == 0) return true;
  return false;
}

bool plausibleDeviceTime(double devMs, double wallMs) {
  if (!(devMs > 0) || !std::isfinite(devMs) || !std::isfinite(wallMs)) return false;
  if (devMs > wallMs * 1.05 + 0.05) return false;
  return devMs >= wallMs * 0.02;
}

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

bool matches(const BackendInfo& i, const std::vector<std::string>& f) { return backendMatches(i.id, f); }



std::string platformName() {
#if defined(__ANDROID__)
  return "android";
#elif defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#else
  return "linux";
#endif
}

std::string jsonEsc(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += char(c);
    }
  }
  return o;
}

std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[48];
  snprintf(b, sizeof b, "%.6g", v);
  return b;
}

std::string statsJson(const Stats& s) {
  return "{\"count\":" + std::to_string(s.count) + ",\"min\":" + num(s.min) + ",\"median\":" + num(s.median) +
         ",\"mean\":" + num(s.mean) + ",\"p95\":" + num(s.p95) + ",\"stddev\":" + num(s.stddev) + "}";
}

}  // namespace

namespace {
BackendInfo sanitizeBackendInfo(BackendInfo bi) {
  bi.id = sanitizeAscii(bi.id);
  bi.api = sanitizeAscii(bi.api);
  bi.device = sanitizeAscii(bi.device);
  bi.vendor = sanitizeAscii(bi.vendor);
  bi.driver = sanitizeAscii(bi.driver);
  bi.note = sanitizeAscii(bi.note);
  std::map<std::string, std::string> clean;
  for (const auto& kv : bi.detail) {
    clean[sanitizeAscii(kv.first)] = sanitizeAscii(kv.second);
  }
  bi.detail = std::move(clean);
  return bi;
}
}  // namespace

Report discoverReport(const DiscoverOptions& opt) {
  HPC_LOGI("[BENCH] discoverReport starting (bigCoresOnly=%d)", opt.cpuBigCoresOnly ? 1 : 0);
  Report r;
  r.version = versionString();
  r.platform = platformName();
  r.cpu = probeCpu();
  auto bs = discoverBackends(opt);
  for (auto& b : bs) {
    if (b) r.backends.push_back(sanitizeBackendInfo(b->info()));
  }
  bs.clear();
  HPC_LOGI("[BENCH] discoverReport finished (%zu backends)", r.backends.size());
  return r;
}

Report runBenchmarks(const BenchConfig& cfg, const ProgressFn& progress, std::atomic<bool>* cancel) {
  HPC_LOGI("[BENCH] runBenchmarks starting: preset=%s warmup=%d iters=%d zeroCopy=%d hetero=%d bigOnly=%d",
           sanitizeAscii(cfg.preset).c_str(), cfg.warmup, cfg.iterations,
           cfg.zeroCopy ? 1 : 0, cfg.hetero ? 1 : 0, cfg.cpuBigOnly ? 1 : 0);
  Report rep;
  rep.version = versionString();
  rep.platform = platformName();
  rep.preset = sanitizeAscii(cfg.preset);
  rep.cpu = probeCpu();
  rep.warmup = cfg.warmup;
  rep.iterations = cfg.iterations;
  auto say = [&](const std::string& s) { if (progress) progress(s); };
  auto cancelled = [&] { return cancel && cancel->load(); };

  DiscoverOptions dopt;
  dopt.cpuBigCoresOnly = cfg.cpuBigOnly;
  say("Discovering backends...");
  auto backends = discoverBackends(dopt);
  for (auto& b : backends) {
    if (b) rep.backends.push_back(sanitizeBackendInfo(b->info()));
  }

  std::vector<Kernel> kernels = cfg.kernels.empty() ? std::vector<Kernel>(std::begin(kAllKernels), std::end(kAllKernels)) : cfg.kernels;
  PrepareOptions popt;
  popt.zeroCopy = cfg.zeroCopy;

  size_t totalJobs = 0;
  for (Kernel k : kernels)
    for (auto& b : backends)
      if (b && b->info().available && matches(b->info(), cfg.backends) && b->supports(k)) ++totalJobs;
  size_t job = 0;

  for (Kernel k : kernels) {
    if (cancelled()) { rep.cancelled = true; break; }
    const uint32_t size = cfg.sizeOverride[int(k)] ? cfg.sizeOverride[int(k)] : presetSize(cfg.preset, k);
    say(std::string("Generating ") + kernelName(k) + " problem (" + std::to_string(size) + ") and fp64 reference...");
    Problem p = makeProblem(k, size);
    const std::vector<double> ref = referenceOutput(p);

    struct Ok { std::string id; double med; };
    std::vector<Ok> okSgemm;

    for (auto& b : backends) {
      if (!b) continue;
      const BackendInfo bi = sanitizeBackendInfo(b->info());
      if (!bi.available || !matches(bi, cfg.backends) || !b->supports(k)) continue;
      if (cancelled()) { rep.cancelled = true; break; }
      ++job;
      const char* kName = kernelName(k);
      say("[" + std::to_string(job) + "/" + std::to_string(totalJobs) + "] " + bi.id + " " + kName + " " + p.sizeLabel());

      KernelResult r;
      r.backendId = bi.id; r.api = bi.api; r.device = bi.device; r.genuine = bi.genuine;
      r.kernel = kName; r.size = p.sizeLabel();
      r.flops = p.flops(); r.bytes = p.bytes();

      const std::string prepBegin = "prepare:" + bi.id + ":" + kName + ":begin";
      const std::string prepEnd = "prepare:" + bi.id + ":" + kName + ":end";
      const std::string runBegin = "run:" + bi.id + ":" + kName + ":begin";
      const std::string runEnd = "run:" + bi.id + ":" + kName + ":end";

      std::string err;
      std::unique_ptr<Execution> ex;
      auto t0 = Clock::now();
      HPC_LOGI("[BENCH] prepare begin: backend=%s kernel=%s size=%s zeroCopy=%d",
               bi.id.c_str(), kName, r.size.c_str(), popt.zeroCopy ? 1 : 0);
      setBreadcrumb(prepBegin);
      try {
        ex = b->prepare(p, popt, &err);
      } catch (const std::exception& e) {
        err = std::string("exception: ") + e.what();
        ex.reset();
      } catch (...) {
        err = "unknown exception in prepare";
        ex.reset();
      }
      setBreadcrumb(prepEnd);
      r.prepareMs = msSince(t0);
      if (!ex) {
        r.error = sanitizeAscii("prepare: " + err);
        HPC_LOGW("[BENCH] prepare failed: backend=%s kernel=%s (%.2f ms): %s",
                 bi.id.c_str(), kName, r.prepareMs, r.error.c_str());
        rep.results.push_back(r);
        rep.allVerified = false;
        continue;
      }
      HPC_LOGI("[BENCH] prepare ok: backend=%s kernel=%s (%.2f ms, zeroCopy=%d)",
               bi.id.c_str(), kName, r.prepareMs, ex->zeroCopy() ? 1 : 0);

      bool good = true;
      std::vector<double> wall, dev;
      size_t devReported = 0, devRejected = 0;
      std::vector<float> out;
      bool readOk = false;
      setBreadcrumb(runBegin);
      try {
        for (int i = 0; i < cfg.warmup && good; ++i) {
          good = ex->run(nullptr, &err);
        }
        for (int i = 0; i < cfg.iterations && good; ++i) {
          double d = -1;
          auto s = Clock::now();
          good = ex->run(&d, &err);
          const double w = msSince(s);
          wall.push_back(w);
          if (d >= 0) {
            ++devReported;
            if (plausibleDeviceTime(d, w)) dev.push_back(d);
            else ++devRejected;
          }
          if (cancelled()) break;
        }
        if (good) {
          readOk = ex->read(out, &err);
        }
      } catch (const std::exception& e) {
        err = std::string("exception: ") + e.what();
        good = false;
        readOk = false;
      } catch (...) {
        err = "unknown exception in run/read";
        good = false;
        readOk = false;
      }
      setBreadcrumb(runEnd);

      if (!good) {
        r.error = sanitizeAscii("run: " + err);
        HPC_LOGW("[BENCH] run failed: backend=%s kernel=%s: %s", bi.id.c_str(), kName, r.error.c_str());
        rep.results.push_back(r);
        rep.allVerified = false;
        continue;
      }
      if (!readOk) {
        r.error = sanitizeAscii("read: " + err);
        HPC_LOGW("[BENCH] read failed: backend=%s kernel=%s: %s", bi.id.c_str(), kName, r.error.c_str());
        rep.results.push_back(r);
        rep.allVerified = false;
        continue;
      }

      const Verdict v = verify(p, ref, out);
      r.ran = true;
      r.verified = v.ok;
      r.maxAbsErr = v.maxAbsErr;
      r.maxRelErr = v.maxRelErr;
      r.zeroCopy = ex->zeroCopy();
      r.wallMs = computeStats(wall);
      // One implausible device sample discredits the timer for the whole run: report none.
      if (devRejected) {
        r.devRejected = devRejected;
        dev.clear();
        HPC_LOGW("[BENCH] %s %s: discarded the device timer (%zu of %zu samples inconsistent with wall time); using wall time",
                 bi.id.c_str(), kName, devRejected, devReported);
      }
      r.devMs = computeStats(dev);
      const bool useDev = r.devMs.count == wall.size() && r.devMs.count > 0 && r.devMs.median > 0;
      const double t = (useDev ? r.devMs.median : r.wallMs.median) * 1e-3;
      r.timeBasis = useDev ? "device" : "wall";
      if (t > 0) { r.gflops = r.flops / t * 1e-9; r.gbps = r.bytes / t * 1e-9; }
      if (!v.ok) {
        rep.allVerified = false;
        r.error = "VERIFICATION FAILED at element " + std::to_string(v.badIndex);
        HPC_LOGE("[BENCH] verification failed: backend=%s kernel=%s badIndex=%zu maxAbsErr=%g",
                 bi.id.c_str(), kName, v.badIndex, v.maxAbsErr);
      } else {
        HPC_LOGI("[BENCH] result ok: backend=%s kernel=%s size=%s wallMed=%.3fms devMed=%.3fms GFLOP/s=%.2f GB/s=%.2f maxAbsErr=%g",
                 bi.id.c_str(), kName, r.size.c_str(), r.wallMs.median, r.devMs.median, r.gflops, r.gbps, r.maxAbsErr);
      }
      // Split partners: the CPU backend itself, and real GPUs/NPUs (not CPU-hosted runtimes such as
      // llvmpipe or PoCL, which would just compete with the CPU half for the same cores).
      if (k == Kernel::Sgemm && v.ok && bi.genuine && (bi.id == "cpu" || bi.deviceClass == "gpu" || bi.deviceClass == "npu"))
        okSgemm.push_back({bi.id, r.wallMs.median});
      rep.results.push_back(r);
    }

    // ---- heterogeneous split: CPU rows + GPU rows of one SGEMM, run concurrently ----
    if (k == Kernel::Sgemm && cfg.hetero && !cancelled() && !rep.cancelled) {
      double cpuMed = 0;
      std::string gpuId;
      double gpuMed = 1e300;
      for (auto& o : okSgemm) {
        if (o.id == "cpu") cpuMed = o.med;
        else if (o.med < gpuMed) { gpuMed = o.med; gpuId = o.id; }
      }
      HeteroResult& h = rep.hetero;
      h.size = p.sizeLabel();
      if (cpuMed <= 0 || gpuId.empty()) {
        h.error = "needs a verified CPU run and a verified GPU/NPU run of sgemm (CPU-hosted runtimes do not count)";
      } else if (p.m < 8) {
        h.error = "problem too small to split (needs >= 8 rows)";
      } else {
        say("Heterogeneous split: cpu + " + gpuId);
        Backend* gpuB = nullptr;
        for (auto& b : backends) if (b && b->info().id == gpuId) gpuB = b.get();
        try {
          // A dedicated CPU backend that leaves one core free for the thread driving the GPU;
          // otherwise pinned CPU workers and the driver thread fight over the same core.
          DiscoverOptions hopt = dopt;
          hopt.cpuReserveCores = 1;
          std::unique_ptr<Backend> cpuOwned = makeCpuBackend(hopt);
          Backend* cpuB = cpuOwned.get();
          const double f = gpuMed / (cpuMed + gpuMed);  // CPU share proportional to its throughput
          uint32_t mc = uint32_t(std::lround(p.m * f / 4.0)) * 4;
          mc = std::min(std::max(mc, 4u), p.m - 4);
          Problem pc = sliceRows(p, 0, mc), pg = sliceRows(p, mc, p.m);
          std::string e1, e2;
          setBreadcrumb("prepare:cpu:sgemm:begin");
          auto cx = cpuB->prepare(pc, popt, &e1);
          setBreadcrumb("prepare:cpu:sgemm:end");
          setBreadcrumb("prepare:" + gpuId + ":sgemm:begin");
          auto gx = gpuB ? gpuB->prepare(pg, popt, &e2) : nullptr;
          setBreadcrumb("prepare:" + gpuId + ":sgemm:end");
          h.gpuId = sanitizeAscii(gpuId); h.cpuRows = mc; h.gpuRows = p.m - mc;
          if (gpuB) h.gpuDevice = sanitizeAscii(gpuB->info().device);
          h.cpuAloneMs = cpuMed; h.gpuAloneMs = gpuMed;
          if (!cx || !gx) {
            rep.allVerified = false;  // an attempted run that did not produce a verified result
            h.error = sanitizeAscii("prepare failed: " + e1 + " " + e2);
            HPC_LOGW("[BENCH] hetero prepare failed: %s", h.error.c_str());
          } else {
            bool good = true; bool c1 = true, g1 = true;
            auto once = [&] {
              std::thread t([&] { c1 = cx->run(nullptr, &e1); });
              g1 = gx->run(nullptr, &e2);
              t.join();
              return c1 && g1;
            };
            setBreadcrumb("run:" + gpuId + ":sgemm:begin");
            for (int i = 0; i < cfg.warmup && good && !cancelled(); ++i) good = once();
            std::vector<double> wall;
            for (int i = 0; i < cfg.iterations && good && !cancelled(); ++i) {
              auto s = Clock::now();
              good = once();
              wall.push_back(msSince(s));
            }
            if (wall.empty()) good = false;  // cancelled before a single timed run
            std::vector<float> oc, og;
            const bool readBoth = good && cx->read(oc, &e1) && gx->read(og, &e2);
            setBreadcrumb("run:" + gpuId + ":sgemm:end");
            if (readBoth) {
              oc.insert(oc.end(), og.begin(), og.end());
              const Verdict v = verify(p, ref, oc);
              h.ran = true; h.verified = v.ok; h.maxAbsErr = v.maxAbsErr;
              h.wallMs = computeStats(wall);
              h.speedupVsBest = std::min(cpuMed, gpuMed) / h.wallMs.median;
              if (!v.ok) rep.allVerified = false;
              HPC_LOGI("[BENCH] hetero sgemm done: cpuRows=%u gpuRows=%u wallMed=%.3fms speedup=%.2fx verified=%d",
                       h.cpuRows, h.gpuRows, h.wallMs.median, h.speedupVsBest, h.verified ? 1 : 0);
            } else if (cancelled()) {
              h.error = "cancelled";
            } else {
              rep.allVerified = false;
              h.error = sanitizeAscii("run/read failed: " + e1 + " " + e2);
              HPC_LOGW("[BENCH] hetero run/read failed: %s", h.error.c_str());
            }
          }
        } catch (const std::exception& e) {
          rep.allVerified = false;
          h.error = sanitizeAscii(std::string("hetero exception: ") + e.what());
          HPC_LOGE("[BENCH] %s", h.error.c_str());
        } catch (...) {
          rep.allVerified = false;
          h.error = "hetero unknown exception";
          HPC_LOGE("[BENCH] %s", h.error.c_str());
        }
      }
    }
  }
  if (cancelled()) rep.cancelled = true;  // also when cancel arrived during the last job
  // Destroy backends on this thread (GL contexts are thread-bound).
  backends.clear();
  HPC_LOGI("[BENCH] runBenchmarks finished (allVerified=%d, cancelled=%d)",
           rep.allVerified ? 1 : 0, rep.cancelled ? 1 : 0);
  return rep;
}

// ---------------------------------------------------------------------------------------------
std::string toJson(const Report& r) {
  std::ostringstream o;
  o << "{\n  \"version\":\"" << jsonEsc(r.version) << "\",\"platform\":\"" << r.platform << "\",\"preset\":\"" << jsonEsc(r.preset)
    << "\",\"warmup\":" << r.warmup << ",\"iterations\":" << r.iterations << ",\"all_verified\":" << (r.allVerified ? "true" : "false")
    << ",\"cancelled\":" << (r.cancelled ? "true" : "false") << ",\n";
  o << "  \"cpu\":{\"arch\":\"" << jsonEsc(r.cpu.arch) << "\",\"cores\":" << r.cpu.logicalCores << ",\"big_cores\":" << r.cpu.bigCores.size()
    << ",\"heterogeneous\":" << (r.cpu.heterogeneous ? "true" : "false") << ",\"topology_readable\":" << (r.cpu.metricReadable ? "true" : "false")
    << ",\"isa\":\"" << jsonEsc(r.cpu.isa) << "\"},\n  \"backends\":[";
  for (size_t i = 0; i < r.backends.size(); ++i) {
    const auto& b = r.backends[i];
    o << (i ? ",\n    " : "\n    ") << "{\"id\":\"" << jsonEsc(b.id) << "\",\"api\":\"" << jsonEsc(b.api) << "\",\"device\":\"" << jsonEsc(b.device)
      << "\",\"vendor\":\"" << jsonEsc(b.vendor) << "\",\"driver\":\"" << jsonEsc(b.driver) << "\",\"device_class\":\"" << jsonEsc(b.deviceClass) << "\",\"available\":" << (b.available ? "true" : "false")
      << ",\"genuine\":" << (b.genuine ? "true" : "false") << ",\"note\":\"" << jsonEsc(b.note) << "\",\"detail\":{";
    bool first = true;
    for (const auto& kv : b.detail) { o << (first ? "" : ",") << "\"" << jsonEsc(kv.first) << "\":\"" << jsonEsc(kv.second) << "\""; first = false; }
    o << "}}";
  }
  o << "\n  ],\n  \"results\":[";
  for (size_t i = 0; i < r.results.size(); ++i) {
    const auto& k = r.results[i];
    o << (i ? ",\n    " : "\n    ") << "{\"backend\":\"" << jsonEsc(k.backendId) << "\",\"device\":\"" << jsonEsc(k.device) << "\",\"genuine\":" << (k.genuine ? "true" : "false")
      << ",\"kernel\":\"" << k.kernel << "\",\"size\":\"" << k.size << "\",\"ran\":" << (k.ran ? "true" : "false") << ",\"verified\":" << (k.verified ? "true" : "false")
      << ",\"error\":\"" << jsonEsc(k.error) << "\",\"prepare_ms\":" << num(k.prepareMs) << ",\"wall_ms\":" << statsJson(k.wallMs) << ",\"device_ms\":" << statsJson(k.devMs)
      << ",\"device_samples_rejected\":" << k.devRejected << ",\"time_basis\":\"" << k.timeBasis << "\",\"gflops\":" << num(k.gflops) << ",\"gbps\":" << num(k.gbps) << ",\"flops\":" << num(k.flops) << ",\"bytes\":" << num(k.bytes)
      << ",\"arithmetic_intensity\":" << num(k.bytes > 0 ? k.flops / k.bytes : 0) << ",\"max_abs_err\":" << num(k.maxAbsErr) << ",\"max_rel_err\":" << num(k.maxRelErr)
      << ",\"zero_copy\":" << (k.zeroCopy ? "true" : "false") << "}";
  }
  const auto& h = r.hetero;
  o << "\n  ],\n  \"hetero\":{\"ran\":" << (h.ran ? "true" : "false") << ",\"error\":\"" << jsonEsc(h.error) << "\",\"kernel\":\"sgemm\",\"size\":\"" << h.size << "\",\"gpu\":\"" << jsonEsc(h.gpuId) << "\",\"gpu_device\":\"" << jsonEsc(h.gpuDevice)
    << "\",\"cpu_rows\":" << h.cpuRows << ",\"gpu_rows\":" << h.gpuRows << ",\"wall_ms\":" << statsJson(h.wallMs) << ",\"cpu_alone_ms\":" << num(h.cpuAloneMs)
    << ",\"gpu_alone_ms\":" << num(h.gpuAloneMs) << ",\"speedup_vs_best_single\":" << num(h.speedupVsBest) << ",\"verified\":" << (h.verified ? "true" : "false")
    << ",\"max_abs_err\":" << num(h.maxAbsErr) << "}\n}\n";
  return o.str();
}

std::string toText(const Report& r) {
  std::ostringstream o;
  char line[320];
  o << "android-hpc " << r.version << " on " << r.platform << " | CPU " << r.cpu.arch << " x" << r.cpu.logicalCores;
  if (r.cpu.metricReadable) o << " (" << r.cpu.bigCores.size() << " big" << (r.cpu.heterogeneous ? ", heterogeneous" : ", homogeneous") << ")";
  else o << " (topology hidden by OS)";
  if (!r.cpu.isa.empty()) o << " | " << r.cpu.isa;
  o << "\n\nBackends\n";
  for (const auto& b : r.backends) {
    o << "  " << (b.available ? "[ok]  " : "[n/a] ") << b.id << "  " << b.api;
    if (!b.device.empty()) o << "  | " << b.device;
    if (!b.genuine) o << "  | NOT GENUINE";
    o << "\n";
    if (!b.note.empty()) o << "        " << b.note << "\n";
    for (const auto& kv : b.detail) o << "        " << kv.first << ": " << kv.second << "\n";
  }
  if (!r.results.empty()) {
    o << "\nResults (preset " << r.preset << ", " << r.warmup << " warmup + " << r.iterations << " timed runs; GFLOP/s and GB/s use device time when the API reports it)\n";
    snprintf(line, sizeof line, "%-10s %-8s %-14s %10s %10s %9s %8s %-5s %s\n", "backend", "kernel", "size", "wall ms", "dev ms", "GFLOP/s", "GB/s", "ok", "max|err|");
    o << line;
    for (const auto& k : r.results) {
      if (!k.ran) { snprintf(line, sizeof line, "%-10s %-8s %-14s  FAILED: %s\n", k.backendId.c_str(), k.kernel.c_str(), k.size.c_str(), k.error.c_str()); o << line; continue; }
      char dev[32] = "-";
      if (k.timeBasis == "device") snprintf(dev, sizeof dev, "%.3f", k.devMs.median);
      snprintf(line, sizeof line, "%-10s %-8s %-14s %10.3f %10s %9.2f %8.2f %-5s %.3g%s\n", k.backendId.c_str(), k.kernel.c_str(), k.size.c_str(), k.wallMs.median,
               dev, k.gflops, k.gbps, k.verified ? "yes" : "NO", k.maxAbsErr, k.genuine ? "" : "  (shim)");
      o << line;
    }
  }
  const auto& h = r.hetero;
  if (h.ran) {
    snprintf(line, sizeof line, "\nHeterogeneous sgemm %s: cpu %u rows + %s %u rows concurrently -> %.3f ms (cpu alone %.3f, %s alone %.3f) speedup vs best single %.2fx, verified %s\n",
             h.size.c_str(), h.cpuRows, h.gpuId.c_str(), h.gpuRows, h.wallMs.median, h.cpuAloneMs, h.gpuId.c_str(), h.gpuAloneMs, h.speedupVsBest, h.verified ? "yes" : "NO");
    o << line;
  } else if (!h.error.empty()) {
    o << "\nHeterogeneous split not run: " << h.error << "\n";
  }
  bool anyDevRejected = false;
  for (const auto& k : r.results) anyDevRejected |= k.devRejected > 0;
  if (anyDevRejected)
    o << "\n\"dev ms\" is '-' where the device timer was inconsistent with wall time and was discarded.\n";
  if (r.cancelled) o << "\nCancelled: the results above are partial.\n";
  o << "\nAll attempted runs verified against fp64 reference: " << (r.allVerified ? "yes" : "NO") << "\n";
  return o.str();
}

}  // namespace hpc
