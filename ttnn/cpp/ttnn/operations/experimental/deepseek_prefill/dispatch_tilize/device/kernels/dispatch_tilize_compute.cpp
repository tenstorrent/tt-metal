// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr bool skip_padding = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_ctl_id = tt::CBIndex::c_1;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<tt::CBIndex::c_0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    uint32_t num_blocks = per_core_block_cnt;
    if constexpr (skip_padding) {
        // this_core_blocks published by the reader (bounded to the filled prefix).
        CircularBuffer cb_ctl(cb_ctl_id);
        cb_ctl.wait_front(1);
        num_blocks = read_tile_value(cb_ctl_id, 0, 0);
    }

    if (num_blocks == 0) {
        return;  // cold core: nothing to tilize
    }

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(num_blocks);
}
