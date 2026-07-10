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

// The reduce packs into cb_acc / cb_out (dst format). When that differs from the input (src) format
// — e.g. bf16 input reduced into an FP32 partial (H-axis-split stage 1) — the packer must also be
// reconfigured, else it writes src-format datums into a dst-format buffer (garbage). The factory
// defines REDUCE_RM_MIXED_FORMAT only in that case, so the common same-dtype path keeps the cheaper
// input-only reconfig.
constexpr auto rm_reconfig_mode =
#ifdef REDUCE_RM_MIXED_FORMAT
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
#else
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT;
#endif

// One reduce() call over the (ht_in_chunk × wt_in_chunk × NC) block currently staged in cb_tile_in.
// is_last_chunk == true packs the final result into cb_out (with optional post-mul); otherwise the
// partial is left in cb_acc at index chunk_idx and accumulation continues on the next call.
FORCE_INLINE void reduce_block(
    uint32_t ht_in_chunk, uint32_t wt_in_chunk, uint32_t NC, uint32_t chunk_idx, bool is_last_chunk) {
    if (is_last_chunk) {
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            cb_tile_in,
            cb_scaler,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            rm_reconfig_mode>(
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
            cb_tile_in,
            cb_scaler,
            cb_acc,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            rm_reconfig_mode>(
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
                const bool is_last_chunk = (wt_base + wt_tiles_per_chunk) >= Wt;

                compute_kernel_lib::tilize<wt_tiles_per_chunk, cb_rm, cb_tile_in>(
                    ht_in_chunk, ht_in_chunk * tt::constants::TILE_HEIGHT);
                reduce_block(ht_in_chunk, wt_tiles_per_chunk, NC, chunk_idx, is_last_chunk);
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
                reduce_block(ht_in_chunk, wt_tiles_per_chunk, NC, chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    }
}
