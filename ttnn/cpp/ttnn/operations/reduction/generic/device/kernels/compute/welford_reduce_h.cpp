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

#ifdef WELFORD_POST_MUL
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

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
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_arg(args::is_std) != 0;
    constexpr auto two_pass_mean_reciprocal = get_arg(args::two_pass_mean_reciprocal);
    constexpr auto two_pass_variance_reciprocal = get_arg(args::two_pass_variance_reciprocal);

    constexpr uint32_t onetile = 1;

    // For FP32 input dfb::in is flagged UnpackToDest by the program factory so copy_tile
    // preserves the FP32 mantissa into DEST for the statistics SFPU consumer. BF16 input: UnpackToSrc.
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    // Destination register indices inside the Tensix DST register file.
    // The statistics LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;
    constexpr uint32_t retained_input_dst = 3;

    // The number of valid rows in the last tile in height dimension.
    // The statistics LLK processes rows naturally, so skip padding rows in the last tile.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        copy_init(dfb::in);
        tile_regs_acquire();
        two_pass_stats_init_shifted();

        for (uint32_t ht = 0; ht < Ht; ++ht) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
            dfb_in.wait_front(ht + 1);
            copy_tile(dfb::in, ht, input_dst);
            constexpr uint32_t stats_input_dst = input_dst;
#else
            dfb_in.wait_front(onetile);
            // Keep var_dst clean: finalization writes only the result rows, so
            // parking pass-one input there would leak stale data into padding.
            const uint32_t stats_input_dst = ht < 2 ? (ht == 0 ? retained_input_dst : mean_dst) : input_dst;
            copy_tile(dfb::in, 0, stats_input_dst);
            dfb_in.pop_front(onetile);
#endif
            if (ht == 0) {
                two_pass_stats_update_shifted_rows<false, true>(
                    stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
            } else {
                two_pass_stats_update_shifted_rows<false>(
                    stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
            }
        }
        two_pass_stats_finish_shifted_mean(two_pass_mean_reciprocal);

#ifdef WELFORD_TWO_PASS_L1_REPLAY
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            copy_tile(dfb::in, ht, input_dst);
            two_pass_stats_update_rows<true>(input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
        }
        dfb_in.pop_front(Ht);
#else
        constexpr uint32_t num_front_retained = Ht < 2 ? Ht : 2;
        for (uint32_t ht = 0; ht < num_front_retained; ++ht) {
            const uint32_t stats_input_dst = ht == 0 ? retained_input_dst : ht;
            two_pass_stats_update_rows<true>(stats_input_dst, 0, ht == Ht - 1 ? last_tile_rows : tile_height);
        }
        if constexpr (Ht > num_front_retained) {
            for (uint32_t ht = num_front_retained; ht < Ht - 1; ++ht) {
                dfb_in.wait_front(onetile);
                copy_tile(dfb::in, 0, retained_input_dst);
                dfb_in.pop_front(onetile);
                two_pass_stats_update_rows<true>(retained_input_dst, 0, tile_height);
            }
            two_pass_stats_update_rows<true>(input_dst, 0, last_tile_rows);
        }
#endif
        two_pass_stats_finalize_to_row(mean_dst, two_pass_variance_reciprocal);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
#ifdef WELFORD_POST_MUL
        binop_with_scalar_tile_init();
        mul_unary_tile(var_dst, post_mul_scaler_bits);
#endif
        tile_regs_commit();

        // Pack variance/std directly to output -- no transpose needed for H reduction
        // because the statistics LLK produces results in row orientation, which matches
        // the desired output layout (one row of results per column of input).
        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out);
        pack_tile(var_dst, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);
    }
}
