// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_tile reader (load_block): streams this core's contiguous tile-rows of x into cb_input_tiles
// on this RISC's NoC, one reserve/push and one read barrier per tile-row. The residual stream rides along
// here (fetch_residual) or is fetched concurrently by the writer RISC on the other NoC; the host picks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(3);
    constexpr bool fetch_residual = get_compile_time_arg_val(4) != 0;
    // Kernel runs DM_DYNAMIC_NOC, so x tiles may alternate NoC0 / NoC1 (per core: split_x RT arg).
    constexpr bool dynamic_noc = get_compile_time_arg_val(5) != 0;
    constexpr auto input_args = TensorAccessorArgs<6>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t core_rows = get_arg_val<uint32_t>(3);
    const bool split_x = dynamic_noc && get_arg_val<uint32_t>(4) != 0;

    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, tile_bytes);

    auto issue_tile_row = [&](const auto& accessor, uint32_t cb, uint32_t first_page, bool split_nocs) {
        cb_reserve_back(cb, tensor_w_tiles);
#ifndef LAYER_NORM_ABLATE_READS  // perf ablation: keep the CB handshake, drop the NoC payload
        uint32_t l1_write_addr = get_write_ptr(cb);
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            const uint8_t noc = split_nocs ? static_cast<uint8_t>(col & 1) : noc_index;
            noc_async_read(accessor.get_noc_addr(first_page + col, 0, noc), l1_write_addr, tile_bytes, noc);
            l1_write_addr += tile_bytes;
        }
#endif
    };
    auto read_barrier = [&]() {
        if constexpr (dynamic_noc) {
            noc_async_read_barrier(0);
            noc_async_read_barrier(1);
        } else {
            noc_async_read_barrier();
        }
    };

    for (uint32_t row = 0; row < core_rows; ++row) {
        const uint32_t first_page = (start_row + row) * tensor_w_tiles;
        issue_tile_row(input_accessor, cb_input_tiles, first_page, split_x);
        if constexpr (fetch_residual) {
            issue_tile_row(residual_accessor, cb_residual_tiles, first_page, false);
        }
        read_barrier();
        cb_push_back(cb_input_tiles, tensor_w_tiles);
        if constexpr (fetch_residual) {
            cb_push_back(cb_residual_tiles, tensor_w_tiles);
        }
    }
}
