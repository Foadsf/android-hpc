// SPDX-License-Identifier: Apache-2.0
#include "hpc/types.hpp"

#include <algorithm>
#include <cmath>

namespace hpc {

const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Sgemm: return "sgemm";
    case Kernel::Conv3x3: return "conv3x3";
    case Kernel::Reduce: return "reduce";
    case Kernel::Triad: return "triad";
    case Kernel::Dense: return "dense";
  }
  return "?";
}

bool parseKernel(const std::string& s, Kernel& out) {
  for (Kernel k : kAllKernels)
    if (s == kernelName(k)) {
      out = k;
      return true;
    }
  return false;
}

size_t Problem::outputCount() const {
  switch (kernel) {
    case Kernel::Sgemm:
    case Kernel::Dense: return size_t(m) * n;
    case Kernel::Conv3x3: return size_t(m) * n;
    case Kernel::Reduce: return 1;
    case Kernel::Triad: return n;
  }
  return 0;
}

double Problem::flops() const {
  const double M = m, K = k, N = n;
  switch (kernel) {
    case Kernel::Sgemm: return 2.0 * M * K * N;
    case Kernel::Dense: return 2.0 * M * K * N + 2.0 * M * N;  // + bias add + relu
    case Kernel::Conv3x3: return 18.0 * std::max(0.0, M - 2) * std::max(0.0, N - 2);
    case Kernel::Reduce: return N;
    case Kernel::Triad: return 2.0 * N;
  }
  return 0;
}

double Problem::bytes() const {
  const double M = m, K = k, N = n;
  switch (kernel) {
    case Kernel::Sgemm: return 4.0 * (M * K + K * N + M * N);
    case Kernel::Dense: return 4.0 * (M * K + K * N + M * N + N);
    case Kernel::Conv3x3: return 4.0 * (2.0 * M * N + 9);
    case Kernel::Reduce: return 4.0 * N;
    case Kernel::Triad: return 12.0 * N;
  }
  return 0;
}

std::string Problem::sizeLabel() const {
  switch (kernel) {
    case Kernel::Sgemm:
    case Kernel::Dense:
      return std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
    case Kernel::Conv3x3: return std::to_string(n) + "x" + std::to_string(m);
    case Kernel::Reduce:
    case Kernel::Triad: return std::to_string(n);
  }
  return "";
}

namespace {
struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
  float next() {  // xorshift32 -> [-1, 1)
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return float(s >> 8) * (1.0f / 8388608.0f) - 1.0f;
  }
};
}  // namespace

Problem makeProblem(Kernel kernel, uint32_t size, uint32_t seed) {
  Problem p;
  p.kernel = kernel;
  Rng r(seed);
  auto fill = [&](std::vector<float>& v, size_t count, bool positive = false) {
    v.resize(count);
    for (auto& x : v) x = positive ? 0.5f * (r.next() + 1.0f) : r.next();
  };
  switch (kernel) {
    case Kernel::Sgemm:
      p.m = p.k = p.n = size;
      fill(p.a, size_t(size) * size);
      fill(p.b, size_t(size) * size);
      break;
    case Kernel::Dense:
      p.m = p.k = p.n = size;
      fill(p.a, size_t(size) * size);
      fill(p.b, size_t(size) * size);
      fill(p.bias, size);
      break;
    case Kernel::Conv3x3:
      p.m = p.n = size;
      fill(p.a, size_t(size) * size);
      fill(p.b, 9);
      break;
    case Kernel::Reduce:
      p.n = size;
      fill(p.a, size, /*positive=*/true);  // positive values keep the sum well-conditioned
      break;
    case Kernel::Triad:
      p.n = size;
      p.alpha = 0.75f;
      fill(p.a, size);
      fill(p.b, size);
      break;
  }
  return p;
}

Problem sliceRows(const Problem& p, uint32_t r0, uint32_t r1) {
  Problem s;
  s.kernel = p.kernel;
  s.m = r1 - r0;
  s.k = p.k;
  s.n = p.n;
  s.alpha = p.alpha;
  s.a.assign(p.a.begin() + size_t(r0) * p.k, p.a.begin() + size_t(r1) * p.k);
  s.b = p.b;
  s.bias = p.bias;
  return s;
}

Stats computeStats(std::vector<double> v) {
  Stats s;
  s.count = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / double(v.size());
  s.p95 = v[std::min(v.size() - 1, size_t(std::ceil(0.95 * double(v.size())) - 1))];
  double var = 0;
  for (double x : v) var += (x - s.mean) * (x - s.mean);
  s.stddev = std::sqrt(var / double(v.size()));
  return s;
}

}  // namespace hpc
