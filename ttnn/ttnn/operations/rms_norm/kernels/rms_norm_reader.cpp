// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NCRISC).
//
// Order (perf_experiments/reader_x_first, measured on the perf-flagged decode shape: x published at
// ~1.55 us instead of ~2.8 us after kernel start):
//   1. load_x_block(0)      — the first DRAM burst is the compute's critical path, so it goes first
//                             (R3: publish_x_shard, the shard is already resident).
//   2. prepare_scaler       — a RISC L1 fill; it runs after the x push, never while NoC data is landing
//                             (L1 arbitration favours the NoC and slows the fill ~6x).
//   3. load_gamma_slice     — consumed only by the last compute phase (scale).
//   4. load_x_block(1..)    — the remaining blocks.
// load_x_block: all `rows * core_w_tiles` tile reads (TILE) or `32 * rows` stick chunks (ROW_MAJOR)
// issued, ONE barrier, ONE push per block (per tile-row for RM).
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
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

// Perf ablation (measurement only, never set in production): RMS_NORM_ABLATE_READ_X stubs the x tile
// reads' payload (the NoC transfer) while keeping the reserve / barrier / push scaffolding and the
// per-tile address generation, so the ablated variant measures issue + sync without the bytes.
#ifndef RMS_NORM_ABLATE_READ_X
#define RMS_NORM_ABLATE_READ_X 0
#endif

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
    const auto prepare_scaler = []() {
        MaybeDeviceZoneScope("reader_scaler");
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_scaler,
            dataflow_kernel_lib::PoolType::SUM,
            dataflow_kernel_lib::ReduceDim::REDUCE_ROW>();
    };

    // ---- load_gamma_slice: this core's [w_tile_start, +core_w_tiles) slice, resident for the kernel ----
    const auto load_gamma_slice = [&]() {
        if constexpr (gamma_mode == 1) {
            // TILE gamma: (1,1,1,W) padded to one tile-row -> tile id == w tile index.
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
            cb_reserve_back(cb_gamma_tiles, core_w_tiles);
            const uint32_t dst = get_write_ptr(cb_gamma_tiles);
            {
                MaybeDeviceZoneScope("reader_gamma_issue");
                for (uint32_t c = 0; c < core_w_tiles; ++c) {
                    noc_async_read(
                        gamma_acc.get_noc_addr(w_tile_start + c), dst + c * gamma_tile_bytes, gamma_tile_bytes);
                }
            }
            {
                MaybeDeviceZoneScope("reader_gamma_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_gamma_tiles, core_w_tiles);
        } else if constexpr (gamma_mode == 2) {
            // RM gamma: one stick. Row 0 of the 32-row stick block is the slice; rows 1..31 are zeroed so
            // the tilize (compute) reads defined data. Only row 0 is ever consumed (BroadcastDim::Row).
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
            const uint32_t chunk_bytes = core_w_tiles * tile_rows * gamma_elem_bytes;
            const uint32_t w_byte_offset = w_tile_start * tile_rows * gamma_elem_bytes;
            cb_reserve_back(cb_gamma_sticks, core_w_tiles);
            {
                MaybeDeviceZoneScope("reader_gamma_zero_fill");
                Noc noc;
                CircularBuffer gamma_sticks(cb_gamma_sticks);
                noc.async_write_zeros(gamma_sticks, (tile_rows - 1) * chunk_bytes, {.offset_bytes = chunk_bytes});
                noc.write_zeros_l1_barrier();
            }
            {
                MaybeDeviceZoneScope("reader_gamma_issue");
                noc_async_read(gamma_acc.get_noc_addr(0, w_byte_offset), get_write_ptr(cb_gamma_sticks), chunk_bytes);
            }
            {
                MaybeDeviceZoneScope("reader_gamma_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_gamma_sticks, core_w_tiles);
        }
    };

    // ---- publish_x_shard (R3): the shard is already resident in this core's L1 — no NoC read.
    // The shard holds shard_w_tiles per row (>= core_w_tiles: a ragged last shard is padded). ----
    if constexpr (sharded) {
        constexpr uint32_t shard_w_tiles = get_named_compile_time_arg_val("SHARD_W_TILES");
        {
            MaybeDeviceZoneScope("reader_publish_shard");
            cb_reserve_back(cb_x_tiles, tensor_row_tiles * shard_w_tiles);
            cb_push_back(cb_x_tiles, tensor_row_tiles * shard_w_tiles);
        }
        prepare_scaler();
        load_gamma_slice();
        return;
    }

    // ---- load_x_block ----
    const auto input_acc = TensorAccessor(input_args, input_addr, in_page_bytes);
    const auto load_x_block = [&](uint32_t block_idx) {
        const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
        const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;
        if constexpr (input_rm) {
            // 32*rows sticks, each this core's byte range of the row; Wc tile pages per tile-row. The helper
            // owns reserve / issue / barrier / push per tile-row, so this zone is the stage's occupancy.
            MaybeDeviceZoneScope("reader_x_sticks");
            dataflow_kernel_lib::read_sticks_for_tilize<cb_x_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_acc,
                tile_rows * rows,
                core_w_tiles * tile_rows * in_elem_bytes,
                tile_rows * block_row_tile_start,
                w_tile_start * tile_rows * in_elem_bytes);
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            {
                MaybeDeviceZoneScope("reader_x_reserve");
                cb_reserve_back(cb_x_tiles, block_tiles);
            }
            uint32_t dst = get_write_ptr(cb_x_tiles);
            {
                MaybeDeviceZoneScope("reader_x_issue");
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t row_base = (block_row_tile_start + r) * tensor_w_tiles + w_tile_start;
                    for (uint32_t c = 0; c < core_w_tiles; ++c) {
                        const uint64_t src = input_acc.get_noc_addr(row_base + c);
                        if constexpr (RMS_NORM_ABLATE_READ_X) {
                            asm volatile("" : : "r"(static_cast<uint32_t>(src)), "r"(dst) : "memory");
                        } else {
                            noc_async_read(src, dst, in_tile_bytes);
                        }
                        dst += in_tile_bytes;
                    }
                }
            }
            {
                MaybeDeviceZoneScope("reader_x_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_x_tiles, block_tiles);
        }
    };

    load_x_block(0);
    prepare_scaler();
    load_gamma_slice();
    for (uint32_t block_idx = 1; block_idx < num_blocks_this_core; ++block_idx) {
        load_x_block(block_idx);
    }
}
