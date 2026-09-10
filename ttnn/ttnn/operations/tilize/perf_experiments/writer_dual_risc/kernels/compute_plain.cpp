// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — PLAIN compute (unchanged across every variant).
// One `compute_kernel_lib::tilize` call per block, InitAndUninit every time
// (amortization is not this bench's concept and is held off in all variants
// for a fair, identical compute cost).
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib::tilize_config;

    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);

    const uint32_t start_block_id = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);
    const uint32_t block_stride = get_arg_val<uint32_t>(2);

    {
        MaybeDeviceZoneScope("compute_startup");
        compute_kernel_hw_startup(cb_input_rows, cb_output_tiles);
    }

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        MaybeDeviceZoneScope("compute_tilize_block");
        compute_kernel_lib::tilize<
            block_width_tiles,
            cb_input_rows,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            Fp32Mode::Fast>(block_row_extent);
    }
}
