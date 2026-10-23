// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (RISCV_1 / NOC_0).
//
// Block operations owned here (op_design.md §Block schedule):
//   prepare_scalers  — once: [full scaler, partial scaler (last_tile_valid_cols)] into cb_reduce_scaler.
//   load_block       — per block of block_rows tile-rows x this core's W slice, pushed per chunk:
//       TILE: block_rows x chunk_w_tiles tiles per push into cb_x_tiles, one read barrier per chunk.
//       RM:   per (tile-row, chunk) one stick slab (32 sticks x chunk_w_tiles*32 elements) per push into
//             cb_x_rm_sticks, one read barrier per slab; compute tilizes it.
//
// Nominal quanta everywhere (mechanism caps M1/M6): every push is the nominal page count even when the
// last block has fewer valid rows or the chunk fewer valid tiles; only the NoC transfers are narrowed.
//
// Sharded placements (Refinement 2):
//   zero-copy (TILE always; RM when the shard is whole tile-rows of whole tiles): cb_x_tiles / cb_x_rm_sticks
//         IS the resident input shard (ttnn.cb_descriptor_from_sharded_tensor); the reader issues no NoC read
//         at all — it publishes the whole shard with one reserve/push.
//   RM restride (any other RM shard): the shard's sticks are local L1; load_block restrides them (local NoC
//         copies, shard stick stride -> chunk_w_tiles*32-element tilize stride, plus the M9 tail zero-fill)
//         into cb_x_rm_sticks. Only the source address changes vs. interleaved: no TensorAccessor read.
//
// Helper deviation (design rejection R-a): the RM stick read is a raw loop, not
// dataflow_kernel_lib::read_sticks_for_tilize, because the ragged last chunk / last W slice must keep the
// nominal L1 stick stride (chunk_w_tiles*32 elements) and the nominal push of chunk_w_tiles pages while
// reading only the valid bytes, and needs the M9 tail zero-fill; the helper derives stride and push count
// from the valid width.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(2);
    constexpr bool is_rm = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(4);
    constexpr uint32_t chunk_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(8);
    // x source: 0 = interleaved (TensorAccessor), 1 = local RM shard restride, 2 = zero-copy (shard backs the CB)
    constexpr uint32_t x_mode = get_compile_time_arg_val(9);
    constexpr bool x_sharded = x_mode != 0;
    constexpr uint32_t shard_stick_bytes = get_compile_time_arg_val(10);  // RM shard: local stick stride
    constexpr auto x_args = TensorAccessorArgs<11>();

    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t chunk_tiles = block_rows * chunk_w_tiles;                  // TILE push quantum (pages)
    constexpr uint32_t stick_stride_bytes = chunk_w_tiles * TILE_W * elem_bytes;  // RM L1 stride per stick
    constexpr uint32_t tile_col_bytes = TILE_W * elem_bytes;                      // bytes of one tile-column in a stick

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t slice_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t slice_w_tiles = get_arg_val<uint32_t>(4);
    const uint32_t last_tile_valid_cols = get_arg_val<uint32_t>(5);
    const uint32_t row_sticks_end = get_arg_val<uint32_t>(6);     // RM: sticks of the row space below this are valid
    const uint32_t slice_valid_elems = get_arg_val<uint32_t>(7);  // valid elements of this core's W slice
    const uint32_t x_col0_bytes = get_arg_val<uint32_t>(8);       // RM: slice start within a source stick

    // prepare_scalers: [full, partial] pair consumed by ReducePartialScaler::with_partial().
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce_scaler,
        dataflow_kernel_lib::PoolType::SUM,
        dataflow_kernel_lib::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce_scaler,
        dataflow_kernel_lib::PoolType::SUM,
        dataflow_kernel_lib::ReduceDim::REDUCE_ROW>(last_tile_valid_cols);

    const auto x_accessor = TensorAccessor(x_args, x_addr);
    const uint32_t num_blocks = (core_rows_tiles + block_rows - 1) / block_rows;

    if constexpr (x_mode == 2) {
        // load_block (zero-copy shard): the shard is already resident in the CB compute reads x from (TILE:
        // cb_x_tiles; RM: cb_x_rm_sticks, the tilize source); publish all of it at once.
        constexpr uint32_t cb_x_source = is_rm ? cb_x_rm_sticks : cb_x_tiles;
        const uint32_t shard_pages = core_rows_tiles * chunks_per_row * chunk_w_tiles;  // short last block: actual
        cb_reserve_back(cb_x_source, shard_pages);
        cb_push_back(cb_x_source, shard_pages);
        return;
    }

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_row0 = block_idx * block_rows;
        const uint32_t remaining_rows = core_rows_tiles - block_row0;
        const uint32_t valid_rows = remaining_rows < block_rows ? remaining_rows : block_rows;

        if constexpr (!is_rm) {
            // load_block (TILE): one push of block_rows x chunk_w_tiles per chunk (contiguous: M5).
            for (uint32_t ci = 0; ci < chunks_per_row; ++ci) {
                const uint32_t chunk_col0 = ci * chunk_w_tiles;
                const uint32_t chunk_valid =
                    chunk_col0 >= slice_w_tiles
                        ? 0
                        : (slice_w_tiles - chunk_col0 < chunk_w_tiles ? slice_w_tiles - chunk_col0 : chunk_w_tiles);
                cb_reserve_back(cb_x_tiles, chunk_tiles);
                const uint32_t l1_base = get_write_ptr(cb_x_tiles);
                const uint32_t tile_bytes = get_tile_size(cb_x_tiles);
                for (uint32_t r = 0; r < valid_rows; ++r) {
                    const uint32_t page_row = (tile_row_start + block_row0 + r) * tensor_w_tiles;
                    uint32_t l1_addr = l1_base + r * chunk_w_tiles * tile_bytes;
                    for (uint32_t col = 0; col < chunk_valid; ++col) {
                        const uint32_t page = page_row + slice_start_tile + chunk_col0 + col;
                        noc_async_read_page(page, x_accessor, l1_addr);
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_x_tiles, chunk_tiles);
            }
        } else {
            // load_block (RM): one 32-stick slab per (tile-row, chunk), compute tilizes each slab.
            for (uint32_t r = 0; r < block_rows; ++r) {
                const uint32_t tile_row = tile_row_start + block_row0 + r;
                const uint32_t stick0 = tile_row * TILE_H;
                uint32_t sticks_valid = 0;
                if (r < valid_rows && stick0 < row_sticks_end) {
                    sticks_valid = row_sticks_end - stick0 < TILE_H ? row_sticks_end - stick0 : TILE_H;
                }
                for (uint32_t ci = 0; ci < chunks_per_row; ++ci) {
                    const uint32_t chunk_elem0 = ci * chunk_w_tiles * TILE_W;  // within the slice
                    uint32_t valid_elems = 0;
                    if (chunk_elem0 < slice_valid_elems) {
                        const uint32_t elems_left = slice_valid_elems - chunk_elem0;
                        valid_elems = chunk_w_tiles * TILE_W < elems_left ? chunk_w_tiles * TILE_W : elems_left;
                    }
                    const uint32_t valid_bytes = valid_elems * elem_bytes;
                    const uint32_t tail_bytes =
                        ((valid_elems + TILE_W - 1) / TILE_W) * tile_col_bytes - valid_bytes;  // M9 zero-fill

                    cb_reserve_back(cb_x_rm_sticks, chunk_w_tiles);
                    uint32_t l1_addr = get_write_ptr(cb_x_rm_sticks);
                    if (valid_bytes > 0) {
                        const uint32_t byte_offset = x_col0_bytes + chunk_elem0 * elem_bytes;
                        for (uint32_t i = 0; i < sticks_valid; ++i) {
                            uint64_t src;
                            if constexpr (x_sharded) {
                                src = get_noc_addr(x_addr + (stick0 + i) * shard_stick_bytes + byte_offset);
                            } else {
                                src = x_accessor.get_noc_addr(stick0 + i, byte_offset);
                            }
                            noc_async_read(src, l1_addr + i * stick_stride_bytes, valid_bytes);
                        }
                        if (tail_bytes > 0) {
                            for (uint32_t i = 0; i < sticks_valid; ++i) {
                                dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                    l1_addr + i * stick_stride_bytes + valid_bytes, tail_bytes, 0);
                            }
                        }
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_x_rm_sticks, chunk_w_tiles);
                }
            }
        }
    }
}
