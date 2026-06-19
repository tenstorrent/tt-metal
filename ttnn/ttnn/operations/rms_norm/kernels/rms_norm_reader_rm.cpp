// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm ROW_MAJOR reader (tilize-wrapped, row-parallel / Regime A style).
//
// Reads row-major sticks (one per (b,c,h) position; the RMS is computed
// independently per stick over its W elements) and feeds them to the unified
// compute for tilize. Chunked along W by `reduce_block` tiles, so the per-core
// L1 footprint is bounded regardless of W.
//
// Refinement 4: gamma is fed ONCE into a resident CB (the unified compute's
// resident-gamma model), matching the TILE readers. TILE gamma is read as Wt
// column tiles (padding tiles to Wt_padded zeroed) directly into cb_gamma_tiled;
// ROW_MAJOR gamma is read as num_chunks chunks of sticks into cb_gamma_rm and
// compute tilizes it once into cb_gamma_tiled.
//
// Native non-aligned handling (no host-side pad/slice):
//   - W non-aligned: the trailing columns of the last real W-tile (and any
//     wholly-synthetic padding tiles) are ZEROED here so they contribute 0 to
//     the per-stick sum-of-squares. inv_W = 1/W (true count) in compute.
//   - H non-aligned: the last tile-block has < 32 valid sticks; the extra
//     padding sticks are zeroed (never written by the writer).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

// Zero [addr, addr+nbytes) via volatile L1 stores. addr must be 4-aligned;
// nbytes is a u32 multiple (callers pass tile-row multiples, which are aligned).
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n = nbytes >> 2;
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 0;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t gamma_is_tile = get_compile_time_arg_val(5);  // gamma.layout == TILE
    constexpr uint32_t Wt = get_compile_time_arg_val(6);             // real tiles along W (ceil(W/32))
    constexpr uint32_t reduce_block = get_compile_time_arg_val(7);   // chunk width in tiles
    constexpr uint32_t num_chunks = get_compile_time_arg_val(8);     // ceil(Wt / reduce_block)
    constexpr uint32_t W = get_compile_time_arg_val(9);              // true element count along W
    constexpr uint32_t in_elem = get_compile_time_arg_val(10);       // input element bytes
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(11);    // gamma element bytes
    constexpr auto input_args = TensorAccessorArgs<12>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t total_sticks = get_arg_val<uint32_t>(4);

    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReduceDim;

    // SUM scaler = 1.0, col-0 (matmul) fill for SUM + REDUCE_ROW.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t in_tile_row_bytes = TILE_W * in_elem;  // bytes of one row inside one tile
    constexpr uint32_t in_padded_chunk_bytes = reduce_block * in_tile_row_bytes;
    constexpr uint32_t gamma_tile_row_bytes = TILE_W * gamma_elem;
    constexpr uint32_t gamma_padded_chunk_bytes = reduce_block * gamma_tile_row_bytes;
    constexpr uint32_t chunk_cols = reduce_block * TILE_W;  // columns spanned by one chunk
    constexpr uint32_t Wt_padded = num_chunks * reduce_block;

    // 2-arg TensorAccessor: page size taken from the tensor's encoded layout
    // (the row-major stick size), NOT the tile size.
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    // ---- gamma read ONCE into a resident CB (unified resident-gamma model) ----
    if constexpr (has_gamma) {
        if constexpr (gamma_is_tile) {
            // gamma is TILE (1,1,1,W) -> Wt column tiles, data in row 0. Read Wt real
            // tiles into cb_gamma_tiled; zero the (Wt_padded - Wt) synthetic padding
            // tiles. Compute holds these resident across all blocks.
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiled);
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
            cb_reserve_back(cb_gamma_tiled, Wt_padded);
            uint32_t l1 = get_write_ptr(cb_gamma_tiled);
            for (uint32_t gt = 0; gt < Wt_padded; ++gt) {
                if (gt < Wt) {
                    noc_async_read_tile(gt, gamma_accessor, l1);
                } else {
                    zero_l1(l1, gamma_tile_bytes);
                }
                l1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiled, Wt_padded);
        } else {
            // gamma is ROW_MAJOR (1,1,1,W): one stick. Stage num_chunks chunks of
            // reduce_block tile-pages (row 0 carries data); compute tilizes once into
            // cb_gamma_tiled. Padding columns are zeroed (kept well-formed).
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t col0 = c * chunk_cols;
                uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
                if (valid_cols > chunk_cols) {
                    valid_cols = chunk_cols;
                }
                const uint32_t chunk_row_bytes = valid_cols * gamma_elem;
                const uint32_t byte_off = col0 * gamma_elem;

                cb_reserve_back(cb_gamma_rm, reduce_block);
                uint32_t l1 = get_write_ptr(cb_gamma_rm);
                if (chunk_row_bytes > 0) {
                    const uint32_t zstart = chunk_row_bytes & ~3u;
                    if (zstart < gamma_padded_chunk_bytes) {
                        zero_l1(l1 + zstart, gamma_padded_chunk_bytes - zstart);
                    }
                    const uint64_t noc_addr = gamma_accessor.get_noc_addr(0, byte_off);
                    noc_async_read(noc_addr, l1, chunk_row_bytes);
                } else {
                    zero_l1(l1, gamma_padded_chunk_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_rm, reduce_block);
            }
        }
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t global_block = start_block + b;
        const uint32_t block_start_stick = global_block * TILE_H;
        uint32_t rows_this_block = total_sticks - block_start_stick;
        if (rows_this_block > TILE_H) {
            rows_this_block = TILE_H;
        }

        // ---- PASS-1 / resident feed: push Wt-rounded chunks of input sticks ----
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t col0 = c * chunk_cols;
            uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
            if (valid_cols > chunk_cols) {
                valid_cols = chunk_cols;
            }
            const uint32_t chunk_row_bytes = valid_cols * in_elem;
            const uint32_t byte_off = col0 * in_elem;

            cb_reserve_back(cb_rm_in, reduce_block);
            uint32_t l1 = get_write_ptr(cb_rm_in);

            for (uint32_t r = 0; r < TILE_H; ++r) {
                if (r < rows_this_block && chunk_row_bytes > 0) {
                    // zero the padding tail (4-aligned start), then read valid bytes on top
                    const uint32_t zstart = chunk_row_bytes & ~3u;
                    if (zstart < in_padded_chunk_bytes) {
                        zero_l1(l1 + zstart, in_padded_chunk_bytes - zstart);
                    }
                    const uint64_t noc_addr = input_accessor.get_noc_addr(block_start_stick + r, byte_off);
                    noc_async_read(noc_addr, l1, chunk_row_bytes);
                } else {
                    // padding stick (H non-aligned) or fully-synthetic padding tile: zero whole row
                    zero_l1(l1, in_padded_chunk_bytes);
                }
                l1 += in_padded_chunk_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm_in, reduce_block);
        }
    }
}
