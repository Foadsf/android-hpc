// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "hpc/cpu.hpp"

namespace hpc {
// Process-wide worker pool shared by the CPU and SYCL-shim backends.
//  bigOnly : use only the fastest cluster(s)
//  reserve : cores left free for other threads (e.g. the thread driving a GPU)
//  pinMode : -1 auto (pin only on heterogeneous CPUs or bigOnly), 0 never, 1 always
ThreadPool& sharedPool(bool bigOnly, unsigned reserve = 0, int pinMode = -1);
}  // namespace hpc
