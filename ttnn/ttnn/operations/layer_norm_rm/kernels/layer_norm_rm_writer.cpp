// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm writer — runs on BRISC.
//
// Per-core work: for each of `num_strips` strips, write each chunk back to
// DRAM by draining cb_output_rm. write_sticks_after_untilize<TILE>
// owns the cb_wait_front / noc_async_write / cb_pop_front cycle.
//
// Each chunk produces 32 sticks of `chunk_bytes` bytes starting at row
// `strip * 32` of the output tensor, offset `c * chunk_bytes` along W.
//
// CT arg layout:
//   [0] BLOCK_SIZE   (unused; reserved for future per-chunk policy hooks)
//   [1] NUM_BLOCKS
//   [2] chunk_bytes
//   [3..] TensorAccessorArgs(output_tensor)
//
// RT arg layout:
//   [0] output_addr
//   [1] num_strips_for_core
//   [2] start_strip_id

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_rm = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(2);
    (void)BLOCK_SIZE;

    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_strips = get_arg_val<uint32_t>(1);
    const uint32_t start_strip = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t strip = start_strip + i;
        const uint32_t strip_start_row = strip * 32;

        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(
                output_accessor,
                /*total_num_rows=*/32,
                /*row_bytes=*/chunk_bytes,
                /*start_page=*/strip_start_row,
                /*byte_offset_within_page=*/c * chunk_bytes);
        }
    }
}
