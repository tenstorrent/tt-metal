// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford H-dimension reduction kernel.
// Reduces along H (rows) directly using the Welford's LLK, which natively reduces rows
// and maintains per-column accumulators.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"

void kernel_main() {
    // Runtime arg: number of independent column-reductions this core must perform.
    // Each column-reduction processes Ht tiles vertically and produces one output tile.
    uint32_t NCWt = get_arg(args::NCWt);

    // Compile-time args:
    // Number of tiles along the H (reduction) dimension.
    constexpr auto Ht = get_arg(args::Ht);
    // The actual number of elements along H (before padding).
    constexpr auto H = get_arg(args::H);
    // Number of elements per tile in the H dimension (typically 32).
    constexpr auto tile_height = get_arg(args::tile_height);
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    const uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
    const bool apply_post_mul = post_mul_scaler_bits != k_identity_scaler_bits;
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    const bool correction = get_arg(args::correction) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    // Buffer that the reader kernel fills with input tiles.
    // For FP32 input dfb::in is flagged UnpackToDest by the program factory so copy_tile
    // preserves the FP32 mantissa into DEST for the welford SFPU consumer. BF16 input: UnpackToSrc.
    DataflowBuffer dfb_in(dfb::in);
    // Buffer where the final variance/std output tile is written.
    DataflowBuffer dfb_out(dfb::out);

    // Destination register indices inside the Tensix DST register file.
    // Welford's LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid rows in the last tile in height dimension.
    // Welford's LLK processes rows naturally, so we skip padding rows
    // in the last tile via welford_update_rows.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        // Welford accumulation along the H dimension for one column of tiles.

        // start_N is the cumulative row count across tiles processed so far; passed
        // to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each row's running-mean update.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        // Process one tile-column along H while keeping a single running Welford state.
        // Welford's running accumulators (mean in LREG4, M2 in LREG5)
        // live in SFPU local registers (LREGs), which are separate
        // from the DST register file.  This means the Welford state survives
        // across tile_regs_release/acquire cycles -- only DST contents are
        // affected by the handshake, not the SFPU accumulators.
        //
        // Only SFPU-compatible operations are used (copy_tile + welford_update), so no
        // configuration conflict exists and the entire loop can run in a single DST window
        // (one acquire before the loop, one commit after the last tile). Only the final
        // iteration needs to expose result tiles to PACK.
        //
        // Per iteration:
        // - For all non-last H tiles, welford_update(input_dst, start_N, ...) consumes the full
        //   tile and updates the running mean/M2 using start_N as the global element offset.
        // - For the last H tile, welford_update_rows(..., last_tile_rows, ...) ignores padded
        //   rows so only valid elements participate in the statistics.
        // - welford_finalize_to_row(mean_dst, scale_idx, ...) converts M2 into variance and
        //   writes final mean/variance tiles into DST.
        // - If is_std, sqrt_tile() turns variance into standard deviation in place.
        // - start_N advances by one tile height each iteration so Welford sees the correct
        //   element count / divisor progression across the whole H reduction.
        copy_init(dfb::in);
        tile_regs_acquire();

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            dfb_in.wait_front(onetile);
            // copy_tile reads dfb::in. For FP32 input it carries UnpackToDest so the
            // FP32 mantissa is preserved into DEST for the welford SFPU consumer.
            copy_tile(dfb::in, 0, input_dst);
            dfb_in.pop_front(onetile);

            if (ht < (Ht - 1)) {
                welford_update<0>(input_dst, start_N, {});
            } else {
                // Last tile: process only valid rows, then finalize
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance conversion:
                //   correction=false: scale_idx = H-1, reciprocal = 1/H  (population variance)
                //   correction=true:  scale_idx = H-2, reciprocal = 1/(H-1) (sample variance)
                const uint32_t scale_idx = correction ? (H - 2) : (H - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                if constexpr (is_std) {
                    sqrt_tile_init();
                    sqrt_tile(var_dst);
                }
                // Apply the user scalar to the reduced output: var(s*x)=s^2 var(x),
                // std(s*x)=|s| std(x). mul_unary_tile is an SFPU op on DEST at full fp32
                // precision .
                if (apply_post_mul) {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(var_dst, post_mul_scaler_bits);
                }
                tile_regs_commit();
            }
            start_N += tile_height;
        }

        // Pack variance/std directly to output -- no transpose needed for H reduction
        // because Welford natively produces results in row orientation which matches
        // the desired output layout (one row of results per column of input).
        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out);
        pack_tile(var_dst, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);
    }
}
