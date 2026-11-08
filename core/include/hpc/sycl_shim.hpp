// SPDX-License-Identifier: Apache-2.0
// A *minimal SYCL-2020-shaped* API (queue, range, id, USM allocation, parallel_for)
// executed on the shared CPU thread pool.
//
// THIS IS NOT A SYCL IMPLEMENTATION. It has no device code generation, no nd_range, no
// local memory, no barriers and no accessors. It exists so the SYCL-style kernel
// source in sycl_backend.cpp can be exercised and benchmarked everywhere; every
// report marks the backend `genuine = false`. A real SYCL toolchain (e.g. AdaptiveCpp)
// is not packaged for Android; see docs/FACT_CHECK.md.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <new>

#include "hpc/cpu.hpp"

namespace hpc {
namespace sycl_shim {

template <int D>
struct id;
template <>
struct id<1> {
  size_t v[1];
  size_t operator[](int) const { return v[0]; }
};
template <>
struct id<2> {
  size_t v[2];
  size_t operator[](int i) const { return v[i]; }
};

template <int D>
struct range;
template <>
struct range<1> {
  size_t r[1];
  explicit range(size_t a) : r{a} {}
  size_t operator[](int) const { return r[0]; }
  size_t size() const { return r[0]; }
};
template <>
struct range<2> {
  size_t r[2];
  range(size_t a, size_t b) : r{a, b} {}
  size_t operator[](int i) const { return r[i]; }
  size_t size() const { return r[0] * r[1]; }
};

class queue {
 public:
  explicit queue(ThreadPool& pool) : pool_(pool) {}

  template <typename F>
  queue& parallel_for(range<1> r, F f) {
    const uint32_t total = uint32_t(r.size());
    pool_.parallelFor(total, grainFor(total), [&](uint32_t b, uint32_t e) {
      for (uint32_t i = b; i < e; ++i) f(id<1>{{i}});
    });
    return *this;
  }

  template <typename F>
  queue& parallel_for(range<2> r, F f) {
    const uint32_t rows = uint32_t(r[0]);
    const size_t cols = r[1];
    pool_.parallelFor(rows, grainFor(rows), [&](uint32_t b, uint32_t e) {
      for (uint32_t i = b; i < e; ++i)
        for (size_t j = 0; j < cols; ++j) f(id<2>{{i, j}});
    });
    return *this;
  }

  void wait() {}  // parallel_for above is synchronous

 private:
  uint32_t grainFor(uint32_t total) const { return std::max(1u, total / (pool_.size() * 8)); }
  ThreadPool& pool_;
};

}  // namespace sycl_shim
}  // namespace hpc
