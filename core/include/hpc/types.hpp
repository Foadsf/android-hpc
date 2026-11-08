// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hpc {

// The five benchmark kernels. All are fp32 on the device under test.
enum class Kernel { Sgemm, Conv3x3, Reduce, Triad, Dense };

inline constexpr Kernel kAllKernels[] = {Kernel::Sgemm, Kernel::Conv3x3, Kernel::Reduce,
                                         Kernel::Triad, Kernel::Dense};

const char* kernelName(Kernel k);
bool parseKernel(const std::string& s, Kernel& out);

// A fully specified problem instance plus its (deterministic) input data.
//
//  Sgemm  : C[m,n] = A[m,k] * B[k,n]
//  Dense  : C[m,n] = relu(A[m,k] * B[k,n] + bias[n])       (one NN layer)
//  Conv3x3: image m rows x n cols, 3x3 weights in b[9]; interior = correlation,
//           1-pixel border copied from the input
//  Reduce : C[0] = sum(a[0..n))
//  Triad  : C[i] = a[i] + alpha * b[i]                      (STREAM triad)
struct Problem {
  Kernel kernel = Kernel::Sgemm;
  uint32_t m = 0, k = 0, n = 0;
  float alpha = 1.0f;
  std::vector<float> a, b, bias;

  size_t outputCount() const;
  double flops() const;  // arithmetic operations (mul+add counted as 2)
  double bytes() const;  // compulsory memory traffic (each operand touched once)
  std::string sizeLabel() const;
};

// `size` is the problem edge for matrix/image kernels and the element count for
// Reduce/Triad. Data is generated from `seed` and is identical on every platform.
Problem makeProblem(Kernel kernel, uint32_t size, uint32_t seed = 1234);

// Row slice [r0, r1) of an Sgemm/Dense problem (used by the heterogeneous split).
Problem sliceRows(const Problem& p, uint32_t r0, uint32_t r1);

struct Stats {
  double min = 0, median = 0, mean = 0, p95 = 0, stddev = 0;
  size_t count = 0;
};
Stats computeStats(std::vector<double> samples);

}  // namespace hpc
