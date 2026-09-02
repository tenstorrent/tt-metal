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
// Buffer layout contract: dfb::rm holds row-sized entries (one entry = one chunk-wide RM row). Per
// (h_chunk, w_chunk) iteration the reader pushes ht_in_chunk * TILE_HEIGHT entries — matching
// compute_kernel_lib::tilize's asymmetric mode (block = 1 tile-row tall, consuming TILE_HEIGHT
// input entries). Padded rows / W columns past valid data carry the reduction identity (0 for SUM)
// from the reader's pre-fill, so they contribute nothing to the running sum.
//
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

namespace {

// Mixed input/output formats (bf16 input, FP32 partial) also need the packer reconfigured; the
// factory defines REDUCE_RM_MIXED_FORMAT only then.
constexpr auto rm_reconfig_mode =
#ifdef REDUCE_RM_MIXED_FORMAT
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
#else
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT;
#endif

// Accurate fp32: enable_fp32_sfpu routes Float32 through the SFPU (full fp32) instead of the
// FPU (tf32).
constexpr auto fp32_mode = get_arg(args::enable_fp32_sfpu) != 0 ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

// One reduce() call over the (ht_in_chunk × wt_in_chunk × NC) block currently staged in dfb::tile_in.
// is_last_chunk == true packs the final result into dfb::out (with optional post-mul); otherwise the
// partial is left in dfb::acc at index chunk_idx and accumulation continues on the next call.
FORCE_INLINE void reduce_block(
    uint32_t ht_in_chunk, uint32_t wt_in_chunk, uint32_t NC, uint32_t chunk_idx, bool is_last_chunk) {
    if (is_last_chunk) {
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            dfb::tile_in,
            dfb::scaler,
            dfb::out,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            rm_reconfig_mode,
            fp32_mode>(
            compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(dfb::acc, chunk_idx),
#ifdef REDUCE_POST_MUL
            [](uint32_t dst_idx) {
                const uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
                if (post_mul_scaler_bits == k_identity_scaler_bits) {
                    return;
                }
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
            dfb::tile_in,
            dfb::scaler,
            dfb::acc,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            rm_reconfig_mode,
            fp32_mode>(
            compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(dfb::acc, chunk_idx),
            compute_kernel_lib::NoOp{});
    }
}

}  // namespace

void kernel_main() {
    // Compile-time args. `Ht` carries different meaning per path: per-core slice for W reduce,
    // total H tiles for H reduce. The factory passes whichever is appropriate.
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto NC = get_arg(args::NC);
    constexpr auto wt_tiles_per_chunk = get_arg(args::wt_tiles_per_chunk);
    constexpr auto ht_tiles_per_chunk = get_arg(args::ht_tiles_per_chunk);
    // args::post_mul_scaler_bits is captured inside reduce_block() under REDUCE_POST_MUL;
    // args::enable_fp32_sfpu (the accurate-fp32 flag) is consumed by fp32_mode above.

    compute_kernel_hw_startup(dfb::rm, dfb::tile_in);

    if constexpr (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) {
        //
        // === W reduce path ===
        //
        // chunk_idx resets per H chunk and advances per W chunk; dfb::acc holds ht_in_chunk partials.
        //
        for (uint32_t ht_base = 0; ht_base < Ht; ht_base += ht_tiles_per_chunk) {
            const uint32_t ht_in_chunk = (ht_base + ht_tiles_per_chunk < Ht) ? ht_tiles_per_chunk : (Ht - ht_base);
            uint32_t chunk_idx = 0;
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
                const bool is_last_chunk = (wt_base + wt_tiles_per_chunk) >= Wt;

                compute_kernel_lib::tilize<wt_tiles_per_chunk, dfb::rm, dfb::tile_in>(
                    ht_in_chunk, ht_in_chunk * tt::constants::TILE_HEIGHT);
                reduce_block(ht_in_chunk, wt_tiles_per_chunk, NC, chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    } else {
        //
        // === H reduce path ===
        //
        // chunk_idx resets per output tile and advances per H chunk; dfb::acc holds wt_tiles_per_chunk
        // (== 1 in current factory) partial tile(s) per output. The second runtime arg
        // (output_tiles_seen) is unused on the compute side now that wt_in_chunk is the compile-time
        // constant; the factory still supplies it, matching the legacy argument list.
        //
        // Only the H factory declares this runtime arg, so only that build has the `args::` token —
        // the reference has to be gated at the preprocessor, since name lookup in the discarded
        // `if constexpr` branch happens regardless of the condition.
#ifdef REDUCE_RM_H_PATH
        const uint32_t num_output_tiles_local = get_arg(args::num_output_tiles_local);
#else
        const uint32_t num_output_tiles_local = 0;  // unreachable: this branch is the H path only
#endif

        constexpr uint32_t Ht_total = Ht;  // For H reduce, the Ht arg IS the total Ht.

        for (uint32_t out_idx = 0; out_idx < num_output_tiles_local; ++out_idx) {
            uint32_t chunk_idx = 0;
            for (uint32_t ht_base = 0; ht_base < Ht_total; ht_base += ht_tiles_per_chunk) {
                const uint32_t ht_in_chunk =
                    (ht_base + ht_tiles_per_chunk < Ht_total) ? ht_tiles_per_chunk : (Ht_total - ht_base);
                const bool is_last_chunk = (ht_base + ht_in_chunk) == Ht_total;

                compute_kernel_lib::tilize<wt_tiles_per_chunk, dfb::rm, dfb::tile_in>(
                    ht_in_chunk, ht_in_chunk * tt::constants::TILE_HEIGHT);
                reduce_block(ht_in_chunk, wt_tiles_per_chunk, NC, chunk_idx, is_last_chunk);
                ++chunk_idx;
            }
        }
    }
}
