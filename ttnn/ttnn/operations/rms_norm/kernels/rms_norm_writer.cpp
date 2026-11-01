// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC / NoC1): exchange_partials_block (k > 1) + store_block.
//
// exchange_partials_block — symmetric unicast all-gather, one round per block:
//   every member m writes its block_rows column-0 partial tiles into slot (r*k + m) of EVERY member's
//   cb_gathered_sumsq (its own included), barriers, then bumps each member's arrival semaphore by 1.
//   The slot base is this core's own cb_gathered write pointer: every member declares the identical CB
//   list and pushes identical counts, so the pointer is the same on all members (lockstep). Depth is 2
//   blocks: a peer can only reach block b+2 after consuming my partial of b+1, which I send only after my
//   compute has popped gathered(b) (exchange(b+1) follows store(b) in this kernel).
//   The semaphore is monotonic: block b is complete at k*(b+1) arrivals; it is never reset.
// store_block
//   TILE: per tile-row, in units of <= out_unit_tiles tiles: wait, write, flush, pop. A unit is also clamped
//         to the cb_out ring end (tracked in out_ring_pos) — its pages are read linearly from the read pointer,
//         and core_w_tiles need not be a multiple of out_unit_tiles.
//   RM  : chunk-major (matches compute's per-chunk untilize): per (chunk, tile-row) one stick-piece write
//         per valid stick; full chunks through write_sticks_after_untilize, the ragged chunk raw.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "rms_norm_common.hpp"

using namespace rms_norm_cb;

constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t H = get_compile_time_arg_val(2);
constexpr uint32_t image_row_tiles = get_compile_time_arg_val(3);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(4);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(6);
constexpr uint32_t w_split_cores = get_compile_time_arg_val(7);
constexpr uint32_t out_unit_tiles = get_compile_time_arg_val(8);
constexpr uint32_t sem_id = get_compile_time_arg_val(9);
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(10);  // cb_out capacity (TILE: out_depth * out_unit)
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(11);  // RT index of the first peer (x, y) pair
constexpr auto out_args = TensorAccessorArgs<12>();

constexpr uint32_t TILE_DIM = 32;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t member_idx = get_arg_val<uint32_t>(8);

    const auto out_acc = TensorAccessor(out_args, out_addr, out_page_bytes);

    [[maybe_unused]] uint32_t sem_addr = 0;
    [[maybe_unused]] uint32_t partial_tile_bytes = 0;
    if constexpr (w_split_cores > 1) {
        sem_addr = get_semaphore(sem_id);
        partial_tile_bytes = get_tile_size(cb_partial_sumsq);
    }

    [[maybe_unused]] uint32_t out_ring_pos = 0;  // TILE: cb_out page position of the read pointer
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;

        if constexpr (w_split_cores > 1) {
            // ---- exchange_partials_block ----
            cb_wait_front(cb_partial_sumsq, rows);
            cb_reserve_back(cb_gathered_sumsq, w_split_cores * rows);
            const uint32_t slot_base = get_write_ptr(cb_gathered_sumsq);
            const uint32_t src = get_read_ptr(cb_partial_sumsq);
            for (uint32_t m = 0; m < w_split_cores; ++m) {
                const uint32_t x = get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * m);
                const uint32_t y = get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * m + 1);
                for (uint32_t r = 0; r < rows; ++r) {
                    noc_async_write(
                        src + r * partial_tile_bytes,
                        get_noc_addr(x, y, slot_base + (r * w_split_cores + member_idx) * partial_tile_bytes),
                        partial_tile_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t m = 0; m < w_split_cores; ++m) {
                const uint32_t x = get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * m);
                const uint32_t y = get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * m + 1);
                noc_semaphore_inc(get_noc_addr(x, y, sem_addr), 1);
            }
            cb_pop_front(cb_partial_sumsq, rows);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), w_split_cores * (blk + 1));
            cb_push_back(cb_gathered_sumsq, w_split_cores * rows);
        }

        // ---- store_block ----
#ifdef RMS_RM
        constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;
        for (uint32_t c = 0; c < chunks_this; ++c) {
            const uint32_t chunk_tiles = (core_w_tiles - c * rm_chunk_w_tiles) < rm_chunk_w_tiles
                                             ? (core_w_tiles - c * rm_chunk_w_tiles)
                                             : rm_chunk_w_tiles;
            const uint32_t col0 = (w_tile_start + c * rm_chunk_w_tiles) * TILE_DIM;
            const uint32_t col_end_tiles = (w_tile_start + c * rm_chunk_w_tiles + chunk_tiles) * TILE_DIM;
            const uint32_t col_end = col_end_tiles < W ? col_end_tiles : W;
            const uint32_t bytes = (col_end - col0) * elem_bytes;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t t = row_start + blk * block_row_tiles + r;
                const uint32_t image = t / image_row_tiles;
                const uint32_t tr = t - image * image_row_tiles;
                const uint32_t first_stick = image * H + tr * TILE_DIM;
                const uint32_t valid_rows = (H - tr * TILE_DIM) < TILE_DIM ? (H - tr * TILE_DIM) : TILE_DIM;
                if (bytes == chunk_row_bytes) {
                    dataflow_kernel_lib::write_sticks_after_untilize<cb_out_rm>(
                        out_acc, valid_rows, bytes, first_stick, col0 * elem_bytes);
                } else {
                    cb_wait_front(cb_out_rm, rm_chunk_w_tiles);
                    const uint32_t l1 = get_read_ptr(cb_out_rm);
                    for (uint32_t s = 0; s < valid_rows; ++s) {
                        noc_async_write(
                            l1 + s * chunk_row_bytes, out_acc.get_noc_addr(first_stick + s, col0 * elem_bytes), bytes);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out_rm, rm_chunk_w_tiles);
                }
            }
        }
#else
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t first_tile = t * tensor_w_tiles + w_tile_start;
            for (uint32_t j0 = 0; j0 < core_w_tiles;) {
                uint32_t n = (core_w_tiles - j0) < out_unit_tiles ? (core_w_tiles - j0) : out_unit_tiles;
                n = (out_cb_pages - out_ring_pos) < n ? (out_cb_pages - out_ring_pos) : n;
                cb_wait_front(cb_out, n);
                const uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_write(l1 + i * out_page_bytes, out_acc.get_noc_addr(first_tile + j0 + i), out_page_bytes);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, n);
                j0 += n;
                out_ring_pos = (out_ring_pos + n == out_cb_pages) ? 0 : out_ring_pos + n;
            }
        }
#endif
    }
    noc_async_write_barrier();
    if constexpr (w_split_cores > 1) {
        noc_async_atomic_barrier();
    }
}
