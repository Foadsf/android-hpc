// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vector>

#include "hpc/types.hpp"

namespace hpc {

// fp64 oracle computed on the CPU. Every backend result is checked against this.
std::vector<double> referenceOutput(const Problem& p);

struct Verdict {
  bool ok = false;
  double maxAbsErr = 0;
  double maxRelErr = 0;   // |err| / max(|ref|, 1e-6)
  size_t badIndex = 0;    // first element outside tolerance (valid when !ok)
};

// Per-kernel tolerance: |out-ref| <= atol + rtol*|ref| for every element.
Verdict verify(const Problem& p, const std::vector<double>& ref, const std::vector<float>& out);

}  // namespace hpc
