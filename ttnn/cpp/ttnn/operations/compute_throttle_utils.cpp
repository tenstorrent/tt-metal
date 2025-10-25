// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/compute_throttle_utils.hpp"

#include <tt-logger/tt-logger.hpp>

namespace ttnn {

namespace operations {

namespace compute_throttle_utils {

void add_stagger_defines_if_needed(
    const tt::ARCH arch, const int num_cores, std::map<std::string, std::string>& mm_kernel_defines) {
    // Empirically deduced di/dt problems appear for matmuls using more than 48
    // cores; when there is 48 cores or less, we never enable stagger since the
    // delay impacts op performance
    constexpr uint32_t WH_B0_MM_MAX_CORES_NO_STAGGER = 48;
    // TODO: determine min core threshold for throttle to be needed on BH
    constexpr uint32_t BH_MM_MAX_CORES_NO_STAGGER = 0;

    // Apply stagger delay on Wormhole B0 on odd rows, so that only half of cores start doing work at once.
    // This is done to mitigate di/dt issues, in case the environment var is set.
    // See issue #9857.
    const char* stagger_type = std::getenv("TT_MM_STAGGER_TYPE");
    const char* stagger_value = std::getenv("TT_MM_STAGGER_VALUE");
    if (stagger_type && ((arch == tt::ARCH::WORMHOLE_B0 && num_cores > WH_B0_MM_MAX_CORES_NO_STAGGER) ||
                         (arch == tt::ARCH::BLACKHOLE && num_cores > BH_MM_MAX_CORES_NO_STAGGER))) {
        // TODO check range for stagger_type
        mm_kernel_defines["MM_STAGGER_TYPE"] = stagger_type;

        if (stagger_value == nullptr) {
            log_warning(tt::LogOp, "Using default stagger value: {}.", 0);
            mm_kernel_defines["MM_STAGGER_VALUE"] = "0";
        } else {
            log_warning(tt::LogOp, "Using stagger value: {}.", stagger_value);
            mm_kernel_defines["MM_STAGGER_VALUE"] = stagger_value;
        }
        log_warning(tt::LogOp, "Stagger type {} enabled for matmul op using {} cores.", stagger_type, num_cores);
    }
}

void throttle_mm_perf(
    tt::ARCH arch, int num_cores, std::map<std::string, std::string>& mm_kernel_defines, ThrottleLevel throttle_level) {
    // Empirically deduced di/dt problems appear for OPs calling matmul using more than 48 cores on WH_B0
    constexpr uint32_t WH_B0_MM_MAX_CORES_NO_THROTTLE = 48;
    // TODO: determine min core threshold for throttle to be needed on BH
    constexpr uint32_t BH_MM_MAX_CORES_NO_THROTTLE = 0;
    const bool mm_throttle_needed = (arch == tt::ARCH::WORMHOLE_B0 && num_cores > WH_B0_MM_MAX_CORES_NO_THROTTLE) ||
                                    (arch == tt::ARCH::BLACKHOLE && num_cores > BH_MM_MAX_CORES_NO_THROTTLE);

    if (!mm_throttle_needed) {
        return;
    }

    // Limit matmul compute throughput by inserting NOP instructions between MVMUL instructions of matmul kernel
    // This will slow down the OP if UNPACK/PACK threads are capable of feeding data sufficiently fast (MATH compute
    // bound)
    const bool mm_throttle_env_enabled = std::getenv("TT_MM_THROTTLE_PERF");

    uint32_t uint_throttle_level = static_cast<uint32_t>(throttle_level);

    // If environment variable is set, this overrides the throttle level parameter
    if (mm_throttle_env_enabled) {
        uint_throttle_level = std::stoi(std::getenv("TT_MM_THROTTLE_PERF"));
    }

    // No throttling requested
    if (uint_throttle_level == 0) {
        return;
    }

    mm_kernel_defines["MM_THROTTLE"] = std::to_string(uint_throttle_level);

    if (uint_throttle_level == 5) {
        log_info(tt::LogOp, "Throttle matmul perf to max 33%");
    } else if (uint_throttle_level == 4) {
        log_info(tt::LogOp, "Throttle matmul perf to max 40%");
    } else if (uint_throttle_level == 3) {
        log_info(tt::LogOp, "Throttle matmul perf to max 50%");
    } else if (uint_throttle_level == 2) {
        log_info(tt::LogOp, "Throttle matmul perf to max 67%");
    } else if (uint_throttle_level == 1) {
        log_info(tt::LogOp, "Throttle matmul perf to max 73%");
    } else {
        mm_kernel_defines["MM_THROTTLE"] = std::to_string(0);
        log_error(
            tt::LogOp,
            "Throttle matmul perf ignored: invalid throttle level {} requested - only {{1,2,3,4,5}} are supported",
            uint_throttle_level);
    }
}

void add_dram_skip_defines_if_needed(
    const tt::ARCH arch, std::map<std::string, std::string>& mm_in1_sender_writer_defines) {
    const bool skip_in1_dram = std::getenv("TT_MM_SKIP_IN1_DRAM");
    if (skip_in1_dram && (arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE)) {
        mm_in1_sender_writer_defines["SKIP_IN1_DRAM"] = "1";
    }
}

bool should_sync_after_in1_dram(const tt::ARCH arch) {
    const bool sync_in1_dram = std::getenv("TT_MM_SYNC_AFTER_IN1_DRAM");
    return sync_in1_dram && (arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE);
}

}  // namespace compute_throttle_utils

}  // namespace operations

}  // namespace ttnn
