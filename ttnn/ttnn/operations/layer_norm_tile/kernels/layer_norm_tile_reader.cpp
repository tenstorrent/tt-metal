// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_tile reader (load_block): streams this core's contiguous tile-rows of x (and residual) into
// cb_input_tiles / cb_residual_tiles, one reserve/push and one read barrier per tile-row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_height = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr bool has_residual = get_compile_time_arg_val(5) != 0;
    constexpr auto input_args = TensorAccessorArgs<6>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(4);

    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, tile_bytes);

    auto read_tile_row = [&](const auto& accessor, uint32_t cb, uint32_t first_page) {
        cb_reserve_back(cb, tensor_w_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb);
#ifndef LAYER_NORM_ABLATE_READS  // perf ablation: keep the CB handshake, drop the NoC payload
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            noc_async_read(accessor.get_noc_addr(first_page + col), l1_write_addr, tile_bytes);
            l1_write_addr += tile_bytes;
        }
#endif
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_rows = (block_idx + 1 == num_blocks) ? last_block_extent : block_height;
        const uint32_t block_first_row = start_row + block_idx * block_height;
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t first_page = (block_first_row + row) * tensor_w_tiles;
            read_tile_row(input_accessor, cb_input_tiles, first_page);
            if constexpr (has_residual) {
                read_tile_row(residual_accessor, cb_residual_tiles, first_page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, tensor_w_tiles);
            if constexpr (has_residual) {
                cb_push_back(cb_residual_tiles, tensor_w_tiles);
            }
        }
    }
}
