// SPDX-License-Identifier: Apache-2.0
// Host test-suite: every available backend must reproduce the fp64 reference on awkward
// (non-multiple-of-tile, non-square) sizes, and the verifier must reject corrupt output.
#include <cmath>
#include <cstdio>
#include <string>

#include "hpc/backend.hpp"
#include "hpc/reference.hpp"

using namespace hpc;

static int failures = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      ++failures;                                         \
      std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

static void testStats() {
  Stats s = computeStats({5, 1, 3, 2, 4});
  CHECK(s.min == 1 && s.median == 3 && std::fabs(s.mean - 3) < 1e-12, "stats basic");
  CHECK(s.p95 == 5, "p95=%g", s.p95);
  Stats e = computeStats({1, 2, 3, 4});
  CHECK(e.median == 2.5, "even median %g", e.median);
}

static void testVerifierRejectsCorruption() {
  Problem p = makeProblem(Kernel::Triad, 1000);
  auto ref = referenceOutput(p);
  std::vector<float> good;
  for (double d : ref) good.push_back(float(d));
  CHECK(verify(p, ref, good).ok, "exact output must pass");
  auto bad = good;
  bad[500] += 0.01f;  // negative control: must reach the gate and be rejected
  Verdict v = verify(p, ref, bad);
  CHECK(!v.ok && v.badIndex == 500, "corrupt element must be rejected (ok=%d idx=%zu)", v.ok, v.badIndex);
  auto nan = good;
  nan[3] = NAN;
  CHECK(!verify(p, ref, nan).ok, "NaN must be rejected");
  auto shortv = good;
  shortv.pop_back();
  CHECK(!verify(p, ref, shortv).ok, "wrong length must be rejected");
}

static void runCase(Backend& b, const Problem& p, const char* label) {
  if (!b.supports(p.kernel)) return;
  std::string err;
  PrepareOptions po;
  auto ex = b.prepare(p, po, &err);
  CHECK(ex != nullptr, "%s %s prepare: %s", b.info().id.c_str(), label, err.c_str());
  if (!ex) return;
  double dev = -1;
  CHECK(ex->run(&dev, &err), "%s %s run: %s", b.info().id.c_str(), label, err.c_str());
  CHECK(ex->run(nullptr, &err), "%s %s second run (re-submit): %s", b.info().id.c_str(), label, err.c_str());
  std::vector<float> out;
  CHECK(ex->read(out, &err), "%s %s read: %s", b.info().id.c_str(), label, err.c_str());
  Verdict v = verify(p, referenceOutput(p), out);
  std::printf("  %-10s %-9s %-16s %s (max|err| %.3g)\n", b.info().id.c_str(), kernelName(p.kernel), label, v.ok ? "ok" : "MISMATCH", v.maxAbsErr);
  CHECK(v.ok, "%s %s mismatch vs fp64 reference at %zu", b.info().id.c_str(), label, v.badIndex);
}

int main() {
  testStats();
  testVerifierRejectsCorruption();

  DiscoverOptions dopt;
  auto backends = discoverBackends(dopt);
  int tested = 0;
  for (auto& b : backends) {
    const auto& i = b->info();
    std::printf("backend %s [%s] %s%s%s\n", i.id.c_str(), i.available ? "available" : "unavailable", i.device.c_str(), i.note.empty() ? "" : " -- ", i.note.c_str());
    if (!i.available) continue;
    ++tested;
    runCase(*b, makeProblem(Kernel::Sgemm, 77), "77^3 (odd)");
    runCase(*b, sliceRows(makeProblem(Kernel::Sgemm, 77), 5, 70), "65x77x77 (nonsq)");
    runCase(*b, makeProblem(Kernel::Dense, 33), "33^3 (odd)");
    runCase(*b, sliceRows(makeProblem(Kernel::Dense, 50), 3, 41), "38x50x50 (nonsq)");
    runCase(*b, makeProblem(Kernel::Conv3x3, 61), "61x61");
    runCase(*b, makeProblem(Kernel::Reduce, 100003), "n=100003");
    runCase(*b, makeProblem(Kernel::Reduce, 1u << 20), "n=2^20");
    runCase(*b, makeProblem(Kernel::Reduce, 7), "n=7");
    runCase(*b, makeProblem(Kernel::Triad, 1001), "n=1001");
    runCase(*b, makeProblem(Kernel::Triad, 3u << 20), "n=3M (grid-stride)");
  }
  CHECK(tested >= 1, "at least the CPU backend must be testable");
  std::printf("\n%s (%d failure%s)\n", failures ? "TESTS FAILED" : "all tests passed", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
