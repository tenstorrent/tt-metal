// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/constants.hpp>
//
// Dense RM reduce compute (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// W reduce path (REDUCE_DIM == REDUCE_ROW):
//   chunk packed Ht and Wt; one tilize pass per W chunk (all H slabs in this H chunk), then one
//   reduce() per W chunk with ReduceInputBlockShape::of(ht_in_chunk, wt_tiles_per_chunk, NC).
//   chunk_idx resets per H chunk and advances per W chunk — accumulator holds ht_in_chunk partial
//   tiles per H chunk.
//
// H reduce path (REDUCE_DIM == REDUCE_COL):
//   each output tile is one work unit; chunk_idx resets per work unit and advances per H chunk.
//   accumulator holds wt_tiles_per_chunk (== 1 in current factory) partial tile(s) per work unit.
//
// CB layout contract: cb_rm holds row-sized pages (one CB page = one chunk-wide RM row). Per
// (h_chunk, w_chunk) iteration the reader pushes ht_in_chunk * TILE_HEIGHT pages — matching
// compute_kernel_lib::tilize's asymmetric mode (block = 1 tile-row tall, consuming TILE_HEIGHT
// input pages). Padded rows / W columns past valid data carry the reduction identity (0 for SUM)
// from the reader's pre-fill, so they contribute nothing to the running sum.
//
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

namespace {

constexpr uint32_t cb_rm = tt::CBIndex::c_24;
constexpr uint32_t cb_tile_in = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_out = tt::CBIndex::c_3;
constexpr uint32_t cb_acc = tt::CBIndex::c_5;

template <uint32_t Index>
FORCE_INLINE void reduce_chunk() {
    using Call = ttnn::kernel_lib::
        BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<7, Index>, cb_tile_in, cb_scaler, cb_out, cb_acc>;
    compute_kernel_lib::reduce<Call>(
#ifdef REDUCE_POST_MUL
        [](uint32_t dst_idx) {
            constexpr auto bits = get_compile_time_arg_val(3);
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_idx, bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
}

FORCE_INLINE void reduce_block(uint32_t chunk_idx, bool is_last_chunk) {
    constexpr uint32_t call_count = get_compile_time_arg_val(6);
    if constexpr (call_count == 1) {
        reduce_chunk<0>();
    } else {
        if (chunk_idx == 0) {
            reduce_chunk<0>();
        } else if (is_last_chunk) {
            reduce_chunk<call_count - 1>();
        } else {
            reduce_chunk<1>();
        }
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
                const bool is_last_chunk = (wt_base + wt_tiles_per_chunk) >= Wt;

                compute_kernel_lib::tilize<wt_tiles_per_chunk, cb_rm, cb_tile_in>(
                    ht_in_chunk, ht_in_chunk * tt::constants::TILE_HEIGHT);
                reduce_block(chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    } else {
        //
        // === H reduce path ===
        //
        // chunk_idx resets per output tile and advances per H chunk; cb_acc holds wt_tiles_per_chunk
        // (== 1 in current factory) partial tile(s) per output. Runtime arg 1 (start_output_tile_id)
        // is unused on the compute side now that wt_in_chunk is the compile-time constant.
        //
        const uint32_t num_output_tiles_local = get_arg_val<uint32_t>(0);

        constexpr uint32_t Ht_total = Ht;  // For H reduce, arg(0) IS the total Ht.

        for (uint32_t out_idx = 0; out_idx < num_output_tiles_local; ++out_idx) {
            uint32_t chunk_idx = 0;
            for (uint32_t ht_base = 0; ht_base < Ht_total; ht_base += ht_tiles_per_chunk) {
                const uint32_t ht_in_chunk =
                    (ht_base + ht_tiles_per_chunk < Ht_total) ? ht_tiles_per_chunk : (Ht_total - ht_base);
                const bool is_last_chunk = (ht_base + ht_in_chunk) == Ht_total;

                compute_kernel_lib::tilize<wt_tiles_per_chunk, cb_rm, cb_tile_in>(
                    ht_in_chunk, ht_in_chunk * tt::constants::TILE_HEIGHT);
                reduce_block(chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    }
}
