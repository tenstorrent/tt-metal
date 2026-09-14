// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NCRISC).
//
// Per core, once:  prepare_scaler (SUM scaler tile = 1.0), load_gamma_slice (this core's W slice of
//                  gamma, resident for the whole kernel), R3: publish_x_shard.
// Per block:       load_x_block — all `rows * core_w_tiles` tile reads (TILE) or `32 * rows` stick
//                  chunks (ROW_MAJOR) issued, ONE barrier, ONE push per block (per tile-row for RM).
//
// Deviations from op_design.md's API mapping (advisory items):
//   * RM x is read with read_sticks_for_tilize<TILE granularity> (Wc tile pages per tile-row, one
//     barrier per 32 sticks) instead of ROW granularity (one barrier per stick); the compute side
//     tilizes in the matching symmetric mode. Same bytes, same L1, 32x fewer NoC barriers.
//   * RM gamma: rows 1..31 of the stick block are zeroed with noc.async_write_zeros (DM engine)
//     rather than a RISC store loop (fill_l1_range); the read of row 0 is then a raw accessor
//     read inside the same reserve/push window (the stick helper's window is not open to a
//     zero-fill, so it cannot be used here).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    // ---- named compile-time args (single source of truth: rms_norm_program_descriptor.py) ----
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("CB_X_TILES");
    constexpr uint32_t cb_x_sticks = get_named_compile_time_arg_val("CB_X_STICKS");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("CB_SCALER");
    constexpr uint32_t cb_gamma_tiles = get_named_compile_time_arg_val("CB_GAMMA_TILES");
    constexpr uint32_t cb_gamma_sticks = get_named_compile_time_arg_val("CB_GAMMA_STICKS");
    constexpr bool input_rm = get_named_compile_time_arg_val("INPUT_RM") != 0;
    constexpr uint32_t gamma_mode = get_named_compile_time_arg_val("GAMMA_MODE");  // 0 none, 1 TILE, 2 RM
    constexpr bool sharded = get_named_compile_time_arg_val("SHARDED") != 0;
    constexpr uint32_t in_page_bytes = get_named_compile_time_arg_val("IN_PAGE_BYTES");
    constexpr uint32_t in_tile_bytes = get_named_compile_time_arg_val("IN_TILE_BYTES");
    constexpr uint32_t in_elem_bytes = get_named_compile_time_arg_val("IN_ELEM_BYTES");
    constexpr uint32_t gamma_page_bytes = get_named_compile_time_arg_val("GAMMA_PAGE_BYTES");
    constexpr uint32_t gamma_tile_bytes = get_named_compile_time_arg_val("GAMMA_TILE_BYTES");
    constexpr uint32_t gamma_elem_bytes = get_named_compile_time_arg_val("GAMMA_ELEM_BYTES");
    constexpr uint32_t tile_rows = 32;

    // ---- positional compile-time args: the two tensor accessors ----
    constexpr auto input_args = TensorAccessorArgs<0>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // ---- runtime args ----
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(3);
    const uint32_t block_rows = get_arg_val<uint32_t>(4);
    const uint32_t last_block_rows = get_arg_val<uint32_t>(5);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(7);
    const uint32_t tensor_w_tiles = get_arg_val<uint32_t>(8);
    const uint32_t tensor_row_tiles = get_arg_val<uint32_t>(9);

    // ---- prepare_scaler: SUM scaler (1.0) in the reduce datapath's own layout ----
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler,
        dataflow_kernel_lib::PoolType::SUM,
        dataflow_kernel_lib::ReduceDim::REDUCE_ROW>();

    // ---- load_gamma_slice: this core's [w_tile_start, +core_w_tiles) slice, resident for the kernel ----
    if constexpr (gamma_mode == 1) {
        // TILE gamma: (1,1,1,W) padded to one tile-row -> tile id == w tile index.
        const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
        cb_reserve_back(cb_gamma_tiles, core_w_tiles);
        const uint32_t dst = get_write_ptr(cb_gamma_tiles);
        for (uint32_t c = 0; c < core_w_tiles; ++c) {
            noc_async_read(gamma_acc.get_noc_addr(w_tile_start + c), dst + c * gamma_tile_bytes, gamma_tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, core_w_tiles);
    } else if constexpr (gamma_mode == 2) {
        // RM gamma: one stick. Row 0 of the 32-row stick block is the slice; rows 1..31 are zeroed so
        // the tilize (compute) reads defined data. Only row 0 is ever consumed (BroadcastDim::Row).
        const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
        const uint32_t chunk_bytes = core_w_tiles * tile_rows * gamma_elem_bytes;
        const uint32_t w_byte_offset = w_tile_start * tile_rows * gamma_elem_bytes;
        cb_reserve_back(cb_gamma_sticks, core_w_tiles);
        {
            Noc noc;
            CircularBuffer gamma_sticks(cb_gamma_sticks);
            noc.async_write_zeros(gamma_sticks, (tile_rows - 1) * chunk_bytes, {.offset_bytes = chunk_bytes});
            noc.write_zeros_l1_barrier();
        }
        noc_async_read(gamma_acc.get_noc_addr(0, w_byte_offset), get_write_ptr(cb_gamma_sticks), chunk_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_sticks, core_w_tiles);
    }

    // ---- publish_x_shard (R3): the shard is already resident in this core's L1 — no NoC read ----
    if constexpr (sharded) {
        cb_reserve_back(cb_x_tiles, tensor_row_tiles * core_w_tiles);
        cb_push_back(cb_x_tiles, tensor_row_tiles * core_w_tiles);
        return;
    }

    // ---- load_x_block per block ----
    const auto input_acc = TensorAccessor(input_args, input_addr, in_page_bytes);
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
        const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;
        if constexpr (input_rm) {
            // 32*rows sticks, each this core's byte range of the row; Wc tile pages per tile-row.
            dataflow_kernel_lib::read_sticks_for_tilize<cb_x_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_acc,
                tile_rows * rows,
                core_w_tiles * tile_rows * in_elem_bytes,
                tile_rows * block_row_tile_start,
                w_tile_start * tile_rows * in_elem_bytes);
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            cb_reserve_back(cb_x_tiles, block_tiles);
            uint32_t dst = get_write_ptr(cb_x_tiles);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row_base = (block_row_tile_start + r) * tensor_w_tiles + w_tile_start;
                for (uint32_t c = 0; c < core_w_tiles; ++c) {
                    noc_async_read(input_acc.get_noc_addr(row_base + c), dst, in_tile_bytes);
                    dst += in_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_tiles, block_tiles);
        }
    }
}
