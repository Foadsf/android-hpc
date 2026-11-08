// SPDX-License-Identifier: Apache-2.0
#include "hpc/reference.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace hpc {

namespace {
template <typename Fn>
void parallelRows(uint32_t rows, Fn fn) {
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  nt = std::min<unsigned>(nt, std::max(1u, rows));
  std::vector<std::thread> th;
  uint32_t per = (rows + nt - 1) / nt;
  for (unsigned t = 0; t < nt; ++t) {
    uint32_t r0 = t * per, r1 = std::min(rows, r0 + per);
    if (r0 >= r1) break;
    th.emplace_back([=] { fn(r0, r1); });
  }
  for (auto& t : th) t.join();
}
}  // namespace

std::vector<double> referenceOutput(const Problem& p) {
  std::vector<double> out(p.outputCount(), 0.0);
  switch (p.kernel) {
    case Kernel::Sgemm:
    case Kernel::Dense: {
      const bool dense = p.kernel == Kernel::Dense;
      parallelRows(p.m, [&](uint32_t r0, uint32_t r1) {
        std::vector<double> row(p.n);
        for (uint32_t i = r0; i < r1; ++i) {
          std::fill(row.begin(), row.end(), 0.0);
          for (uint32_t kk = 0; kk < p.k; ++kk) {
            const double av = p.a[size_t(i) * p.k + kk];
            const float* br = &p.b[size_t(kk) * p.n];
            for (uint32_t j = 0; j < p.n; ++j) row[j] += av * double(br[j]);
          }
          for (uint32_t j = 0; j < p.n; ++j) {
            double v = row[j];
            if (dense) v = std::max(0.0, v + double(p.bias[j]));
            out[size_t(i) * p.n + j] = v;
          }
        }
      });
      break;
    }
    case Kernel::Conv3x3: {
      parallelRows(p.m, [&](uint32_t r0, uint32_t r1) {
        for (uint32_t y = r0; y < r1; ++y)
          for (uint32_t x = 0; x < p.n; ++x) {
            const size_t idx = size_t(y) * p.n + x;
            if (y == 0 || x == 0 || y == p.m - 1 || x == p.n - 1) {
              out[idx] = p.a[idx];
              continue;
            }
            double s = 0;
            for (int dy = 0; dy < 3; ++dy)
              for (int dx = 0; dx < 3; ++dx)
                s += double(p.a[size_t(y + dy - 1) * p.n + (x + dx - 1)]) * double(p.b[dy * 3 + dx]);
            out[idx] = s;
          }
      });
      break;
    }
    case Kernel::Reduce: {
      long double s = 0;
      for (float v : p.a) s += v;
      out[0] = double(s);
      break;
    }
    case Kernel::Triad:
      for (uint32_t i = 0; i < p.n; ++i) out[i] = double(p.a[i]) + double(p.alpha) * double(p.b[i]);
      break;
  }
  return out;
}

Verdict verify(const Problem& p, const std::vector<double>& ref, const std::vector<float>& out) {
  Verdict v;
  if (out.size() != ref.size()) return v;
  double atol = 1e-3, rtol = 1e-4;
  switch (p.kernel) {
    case Kernel::Sgemm:
    case Kernel::Dense: atol = 1e-3; rtol = 1e-4; break;  // K up to a few thousand, |a|,|b|<=1
    case Kernel::Conv3x3: atol = 1e-5; rtol = 1e-5; break;
    case Kernel::Reduce: atol = 0; rtol = 2e-6; break;  // fp32 tree/chunk sums stay within a few ulp
    case Kernel::Triad: atol = 1e-6; rtol = 1e-6; break;
  }
  v.ok = true;
  bool first = true;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double r = ref[i], o = double(out[i]);
    const double e = std::fabs(o - r);
    const bool finite = std::isfinite(o);
    v.maxAbsErr = std::max(v.maxAbsErr, finite ? e : 1e30);
    v.maxRelErr = std::max(v.maxRelErr, finite ? e / std::max(std::fabs(r), 1e-6) : 1e30);
    if (!finite || e > atol + rtol * std::fabs(r)) {
      if (first) v.badIndex = i;
      first = false;
      v.ok = false;
    }
  }
  return v;
}

}  // namespace hpc
