// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <map>

#include "umd/device/types/arch.h"

namespace ttnn {

namespace operations {

namespace compute_throttle_utils {

void add_stagger_defines_if_needed(
    const tt::ARCH arch, const int num_cores, std::map<std::string, std::string>& mm_kernel_defines);
void add_dram_skip_defines_if_needed(
    const tt::ARCH arch, std::map<std::string, std::string>& mm_in1_sender_writer_defines);
bool should_sync_after_in1_dram(const tt::ARCH arch);

/*
 * Optionally limit matmul compute throughput by inserting NOP instructions between MVMUL instructions of matmul kernel
 * This will slow down the OP if UNPACK/PACK threads are capable of feeding data sufficiently fast (MATH compute bound)
 *
 * Enabled by setting env var TT_MM_THROTTLE_PERF to value in range {1,2,3,4,5}
 * Each value corresponds to level of throttling as:
 * Level 1: throttle to 73% of max
 * Level 2: throttle to 67% of max
 * Level 3: throttle to 50% of max
 * Level 4: throttle to 40% of max
 * Level 5: throttle to 33% of max
 */
void throttle_mm_perf(const tt::ARCH arch, const int num_cores, std::map<std::string, std::string>& mm_kernel_defines);

}  // namespace compute_throttle_utils

}  // namespace operations

}  // namespace ttnn
