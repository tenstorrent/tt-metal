// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ostream>
#include <string>
#include <string_view>
#include <map>

#include <umd/device/types/arch.hpp>

namespace ttnn::operations::compute_throttle_utils {

enum class ThrottleLevel : uint32_t {
    NO_THROTTLE = 0,
    LEVEL_1 = 1,
    LEVEL_2 = 2,
    LEVEL_3 = 3,
    LEVEL_4 = 4,
    LEVEL_5 = 5
};

inline std::ostream& operator<<(std::ostream& os, ThrottleLevel level) {
    switch (level) {
        case ThrottleLevel::NO_THROTTLE: return os << "ThrottleLevel::NO_THROTTLE";
        case ThrottleLevel::LEVEL_1: return os << "ThrottleLevel::LEVEL_1";
        case ThrottleLevel::LEVEL_2: return os << "ThrottleLevel::LEVEL_2";
        case ThrottleLevel::LEVEL_3: return os << "ThrottleLevel::LEVEL_3";
        case ThrottleLevel::LEVEL_4: return os << "ThrottleLevel::LEVEL_4";
        case ThrottleLevel::LEVEL_5: return os << "ThrottleLevel::LEVEL_5";
        default: return os << "ThrottleLevel(" << static_cast<uint32_t>(level) << ")";
    }
}

/*
 * Optionally delay the odd-row cores of a matmul before they start on in1, so that not all cores draw current at
 * once (di/dt). Enabled by the TT_MM_STAGGER_TYPE (1, 2 or 3) and TT_MM_STAGGER_VALUE (cycles) env vars; adds the
 * MM_STAGGER_TYPE / MM_STAGGER_VALUE / MM_STAGGER_OPERAND defines the compute kernel's llk_wait_tiles hook reads.
 *
 * in1_operand is the compute kernel's own name for in1's operand id, pasted into MM_STAGGER_OPERAND verbatim: the CB
 * index as text for a circular-buffer kernel (std::to_string(src1_cb_index)), or dfb_slot_operand(<accessor>) for a
 * Metal 2.0 kernel, since a DFB's device slot is assigned when the program is built and is not always 1.
 */
void add_stagger_defines_if_needed(
    tt::ARCH arch,
    int num_cores,
    std::map<std::string, std::string>& mm_kernel_defines,
    const std::string& in1_operand);

// The in1 operand for add_stagger_defines_if_needed when the compute kernel is a Metal 2.0 kernel whose in1 consumer
// DFB is bound under `accessor_name`: the DFB_SLOT_<name> constant the generated kernel bindings header defines for
// every DFB binding. The dfb::<name> token itself cannot serve, as that header declares it only after it has pulled
// in the LLK io headers that read MM_STAGGER_OPERAND.
inline std::string dfb_slot_operand(std::string_view accessor_name) { return "DFB_SLOT_" + std::string(accessor_name); }
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
    ThrottleLevel throttle_level = ThrottleLevel::NO_THROTTLE);

}  // namespace ttnn::operations::compute_throttle_utils
