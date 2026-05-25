// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
//
// Dense RM reduce compute (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// W reduce path (REDUCE_DIM == REDUCE_ROW):
//   chunk packed Ht and Wt; one tilize pass per W chunk (all H slabs in this H chunk), then one reduce()
//   per W chunk with ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC). chunk_idx resets per H chunk
//   and advances per W chunk — accumulator holds ht_in_chunk partial tiles per H chunk.
//
// H reduce path (REDUCE_DIM == REDUCE_COL):
//   each (nc, w_chunk) tuple is a work unit; chunk_idx resets per work unit and advances per H chunk.
//   accumulator holds wt_in_chunk partial tiles per work unit. The host passes ht/wt chunk sizes plus
//   per-core (num_output_tiles_local, start_output_tile_id) as runtime args.
//
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

namespace {

constexpr uint32_t cb_rm = tt::CBIndex::c_24;
constexpr uint32_t cb_tile_in = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_out = tt::CBIndex::c_3;
constexpr uint32_t cb_acc = tt::CBIndex::c_5;

// Tilize ht_in_chunk slab pages (each carrying wt_in_chunk tiles of packed RM rows) into cb_tile_in.
// Caller is responsible for issuing the reduce() that drains cb_tile_in.
FORCE_INLINE void tilize_chunk(uint32_t ht_in_chunk, uint32_t wt_in_chunk) {
    tilize_init(cb_rm, wt_in_chunk, cb_tile_in);
    for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
        cb_wait_front(cb_rm, 1);
        cb_reserve_back(cb_tile_in, wt_in_chunk);
        tilize_block(cb_rm, wt_in_chunk, cb_tile_in);
        cb_pop_front(cb_rm, 1);
        cb_push_back(cb_tile_in, wt_in_chunk);
    }
    tilize_uninit(cb_rm, cb_tile_in);
}

// One reduce() call over the (ht_in_chunk × wt_in_chunk × NC) block currently staged in cb_tile_in.
// is_last_chunk == true packs the final result into cb_out (with optional post-mul); otherwise the partial
// is left in cb_acc at index chunk_idx and accumulation continues on the next call.
FORCE_INLINE void reduce_block(
    uint32_t ht_in_chunk, uint32_t wt_in_chunk, uint32_t NC, uint32_t chunk_idx, bool is_last_chunk) {
    if (is_last_chunk) {
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            cb_tile_in,
            cb_scaler,
            cb_out,
            compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx),
#ifdef REDUCE_POST_MUL
            [](uint32_t dst_idx) {
                constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
                binop_with_scalar_tile_init();
                mul_unary_tile(dst_idx, post_mul_scaler_bits);
            }
#else
            compute_kernel_lib::NoOp{}
#endif
        );
    } else {
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            cb_tile_in,
            cb_scaler,
            cb_acc,
            compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx),
            compute_kernel_lib::NoOp{});
    }
}

}  // namespace

void kernel_main() {
    // Compile-time args. `Ht` carries different meaning per path: per-core slice for W reduce,
    // total H tiles for H reduce. The factory passes whichever is appropriate.
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(4);
    constexpr uint32_t ht_tiles_per_chunk = get_compile_time_arg_val(5);
    // arg(3) = post_mul_scaler_bits — captured inside reduce_block() under REDUCE_POST_MUL.

    compute_kernel_hw_startup(cb_rm, cb_tile_in);

    if constexpr (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) {
        //
        // === W reduce path ===
        //
        // chunk_idx resets per H chunk and advances per W chunk; cb_acc holds ht_in_chunk partials.
        //
        for (uint32_t ht_base = 0; ht_base < Ht; ht_base += ht_tiles_per_chunk) {
            const uint32_t ht_in_chunk = (ht_base + ht_tiles_per_chunk < Ht) ? ht_tiles_per_chunk : (Ht - ht_base);
            uint32_t chunk_idx = 0;
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
                const uint32_t wt_in_chunk = (wt_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - wt_base);
                const bool is_last_chunk = (wt_base + wt_in_chunk) == Wt;

                tilize_chunk(ht_in_chunk, wt_in_chunk);
                reduce_block(ht_in_chunk, wt_in_chunk, NC, chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    } else {
        //
        // === H reduce path ===
        //
        // chunk_idx resets per work unit (one (nc, w_chunk) tuple) and advances per H chunk; cb_acc holds
        // wt_in_chunk partials per work unit. Runtime args are decomposed into (nc, wt_in_nc) tuples.
        //
        const uint32_t num_output_tiles_local = get_arg_val<uint32_t>(0);
        const uint32_t start_output_tile_id = get_arg_val<uint32_t>(1);

        constexpr uint32_t Ht_total = Ht;  // For H reduce, arg(0) IS the total Ht.
        uint32_t outputs_remaining = num_output_tiles_local;
        uint32_t wt_in_nc = start_output_tile_id % Wt;

        while (outputs_remaining > 0) {
            // Largest chunk we can take without crossing an NC boundary or exceeding remaining work.
            uint32_t wt_in_chunk = wt_tiles_per_chunk;
            if (wt_in_chunk > Wt - wt_in_nc) {
                wt_in_chunk = Wt - wt_in_nc;
            }
            if (wt_in_chunk > outputs_remaining) {
                wt_in_chunk = outputs_remaining;
            }

            uint32_t chunk_idx = 0;
            for (uint32_t ht_base = 0; ht_base < Ht_total; ht_base += ht_tiles_per_chunk) {
                const uint32_t ht_in_chunk =
                    (ht_base + ht_tiles_per_chunk < Ht_total) ? ht_tiles_per_chunk : (Ht_total - ht_base);
                const bool is_last_chunk = (ht_base + ht_in_chunk) == Ht_total;

                tilize_chunk(ht_in_chunk, wt_in_chunk);
                reduce_block(ht_in_chunk, wt_in_chunk, NC, chunk_idx, is_last_chunk);
                ++chunk_idx;
            }

            wt_in_nc += wt_in_chunk;
            outputs_remaining -= wt_in_chunk;
            if (wt_in_nc == Wt) {
                wt_in_nc = 0;
            }
        }
    }
}
