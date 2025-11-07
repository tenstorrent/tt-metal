// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <map>

#include <umd/device/types/arch.hpp>

namespace ttnn {

namespace operations {

namespace compute_throttle_utils {

// Empirically deduced di/dt problems appear for matmuls using more than 48
// cores; when there is 48 cores or less, we never enable throttle or stagger since the
// delay impacts op performance
constexpr uint32_t WH_B0_MM_MAX_CORES_NO_THROTTLE_OR_STAGGER = 48;

// currently not defined, so leave at 0.
// Todo: determine min core threshold for throttle/stagger to be needed on BH
constexpr uint32_t BH_MM_MAX_CORES_NO_THROTTLE_OR_STAGGER = 0;

enum class ThrottleLevel : uint32_t {
    NO_THROTTLE = 0,
    LEVEL_1 = 1,
    LEVEL_2 = 2,
    LEVEL_3 = 3,
    LEVEL_4 = 4,
    LEVEL_5 = 5
};

void add_stagger_defines_if_needed(tt::ARCH arch, int num_cores, std::map<std::string, std::string>& mm_kernel_defines);
void add_dram_skip_defines_if_needed(tt::ARCH arch, std::map<std::string, std::string>& mm_in1_sender_writer_defines);
bool should_sync_after_in1_dram(tt::ARCH arch);

/*
 * Optionally limit matmul compute throughput by inserting NOP instructions between MVMUL instructions of matmul kernel
 * This will slow down the OP if UNPACK/PACK threads are capable of feeding data sufficiently fast (MATH compute bound)
 *
 * Enabled by setting env var TT_MM_THROTTLE_PERF to value in range {1,2,3,4,5}
 * If a global env var is not set, the throttle_level parameter is used.
 * Each value corresponds to level of throttling as:
 * Level 1: throttle to 73% of max
 * Level 2: throttle to 67% of max
 * Level 3: throttle to 50% of max
 * Level 4: throttle to 40% of max
 * Level 5: throttle to 33% of max
 */
void throttle_mm_perf(
    tt::ARCH arch,
    int num_cores,
    std::map<std::string, std::string>& mm_kernel_defines,
    uint32_t out_subblock_h_ntiles,
    uint32_t out_subblock_w_ntiles,
    ThrottleLevel throttle_level = ThrottleLevel::NO_THROTTLE);

}  // namespace compute_throttle_utils

}  // namespace operations

}  // namespace ttnn
