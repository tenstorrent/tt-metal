// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_inflight bake-off compute — HELD CONSTANT across every arm.
// One library tilize call for the whole core (identical to the op's compute
// kernel), so any measured delta is attributable to the reader alone.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<
        wt_chunk,
        cb_in,
        cb_out,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure,
        Fp32Mode::Fast>(num_blocks);
}
