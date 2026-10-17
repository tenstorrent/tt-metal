// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/compute_throttle_utils.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::compute_throttle_utils {

namespace {

// Parses the whole of `text` as an unsigned decimal integer; anything else (empty, sign, trailing characters) fails.
bool parse_env_uint(std::string_view text, uint32_t& value) {
    const char* end = text.data() + text.size();
    auto [ptr, ec] = std::from_chars(text.data(), end, value);
    return !text.empty() && ec == std::errc() && ptr == end;
}

}  // namespace

void add_stagger_defines_if_needed(
    const tt::ARCH arch,
    const int num_cores,
    std::map<std::string, std::string>& mm_kernel_defines,
    const std::string& in1_operand) {
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
        // apply_mm_stagger (llk_io_unpack.h) selects its arm with `#if MM_STAGGER_TYPE == 1/2/3` and rejects any
        // other defined type with #error; checking here turns that kernel build failure into a host error that names
        // the env var. The type codes are the ones that #error and the kernel arms list.
        constexpr uint32_t min_stagger_type = 1;
        constexpr uint32_t max_stagger_type = 3;
        // The kernel stores the value in a constexpr int and spins that many cycles.
        constexpr uint32_t max_stagger_cycles = std::numeric_limits<int32_t>::max();
        uint32_t stagger_type_number = 0;
        TT_FATAL(
            parse_env_uint(stagger_type, stagger_type_number) && stagger_type_number >= min_stagger_type &&
                stagger_type_number <= max_stagger_type,
            "TT_MM_STAGGER_TYPE must be {} (first block), 2 (every block) or {} (periodic), got '{}'",
            min_stagger_type,
            max_stagger_type,
            stagger_type);
        // The defines are pasted into the kernel as C++ literals, so re-emit the parsed numbers: an env value with a
        // leading zero ("0100") is decimal here but would be an octal literal (or, for "08", a build error) there.
        mm_kernel_defines["MM_STAGGER_TYPE"] = std::to_string(stagger_type_number);
        mm_kernel_defines["MM_STAGGER_OPERAND"] = in1_operand;

        if (stagger_value == nullptr) {
            log_warning(tt::LogOp, "Using default stagger value: {}.", 0);
            mm_kernel_defines["MM_STAGGER_VALUE"] = "0";
        } else {
            uint32_t stagger_cycles = 0;
            TT_FATAL(
                parse_env_uint(stagger_value, stagger_cycles) && stagger_cycles <= max_stagger_cycles,
                "TT_MM_STAGGER_VALUE must be a number of cycles between 0 and {}, got '{}'",
                max_stagger_cycles,
                stagger_value);
            log_warning(tt::LogOp, "Using stagger value: {}.", stagger_cycles);
            mm_kernel_defines["MM_STAGGER_VALUE"] = std::to_string(stagger_cycles);
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
    const char* mm_throttle_env = std::getenv("TT_MM_THROTTLE_PERF");

    uint32_t uint_throttle_level = static_cast<uint32_t>(throttle_level);

    // If environment variable is set, this overrides the throttle level parameter
    if (mm_throttle_env) {
        uint_throttle_level = std::stoi(mm_throttle_env);
    }

    // No throttling requested
    if (uint_throttle_level == 0) {
        return;
    }

    mm_kernel_defines["MM_THROTTLE"] = std::to_string(uint_throttle_level);

    if (uint_throttle_level == 5) {
        log_debug(tt::LogOp, "Throttle matmul perf to max 33%");
    } else if (uint_throttle_level == 4) {
        log_debug(tt::LogOp, "Throttle matmul perf to max 40%");
    } else if (uint_throttle_level == 3) {
        log_debug(tt::LogOp, "Throttle matmul perf to max 50%");
    } else if (uint_throttle_level == 2) {
        log_debug(tt::LogOp, "Throttle matmul perf to max 67%");
    } else if (uint_throttle_level == 1) {
        log_debug(tt::LogOp, "Throttle matmul perf to max 73%");
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

}  // namespace ttnn::operations::compute_throttle_utils
