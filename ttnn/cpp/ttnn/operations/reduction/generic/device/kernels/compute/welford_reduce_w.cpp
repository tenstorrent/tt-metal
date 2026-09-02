// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef WELFORD_POST_MUL
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Runtime args:
    // Total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    uint32_t NCHt = get_arg(args::NCHt);

    // Compile-time args:
    // Number of tiles along the W (reduction) dimension.
    constexpr auto Wt = get_arg(args::Wt);
    // The actual number of elements along W (before tiling).
    constexpr auto W = get_arg(args::W);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr auto tile_width = get_arg(args::tile_width);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    // For FP32 input dfb::in is flagged UnpackToDest by the program factory so the statistics SFPU
    // intake (transpose_tile) reads with full FP32 precision. For BF16 input it is UnpackToSrc.
    constexpr auto two_pass_mean_reciprocal = get_arg(args::two_pass_mean_reciprocal);
    constexpr auto two_pass_variance_reciprocal = get_arg(args::two_pass_variance_reciprocal);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_var(dfb::var);

    // Destination register indices inside the Tensix DST register file.
    // The statistics LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current transposed input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;
    constexpr uint32_t retained_input_dst = 3;

    // The number of valid columns in the last tile in width dimension.
    // Because the statistics LLK is given transposed data, skip some rows when
    // we want to skip some columns from getting processed.
    constexpr uint32_t last_tile_rows = ((W % tile_width) == 0) ? tile_width : (W % tile_width);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        reconfig_data_format_srca(dfb::in);
        transpose_init(dfb::in);
        tile_regs_acquire();
        two_pass_stats_init_shifted();

        for (uint32_t wt = 0; wt < Wt; ++wt) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
            // Do not pop pass-one tiles: the enlarged CB holds the complete
            // reduction row so pass two can index the same L1 pages directly.
            dfb_in.wait_front(wt + 1);
            transpose_tile(dfb::in, wt, input_dst);
            constexpr uint32_t stats_input_dst = input_dst;
#else
            dfb_in.wait_front(onetile);
            const uint32_t stats_input_dst = wt < 2 ? (wt == 0 ? retained_input_dst : mean_dst) : input_dst;
            transpose_tile(dfb::in, 0, stats_input_dst);
            dfb_in.pop_front(onetile);
#endif
            if (wt == 0) {
                two_pass_stats_update_shifted_rows<false, true>(
                    stats_input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
            } else {
                two_pass_stats_update_shifted_rows<false>(
                    stats_input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
            }
        }
        two_pass_stats_finish_shifted_mean(two_pass_mean_reciprocal);

#ifdef WELFORD_TWO_PASS_L1_REPLAY
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            transpose_tile(dfb::in, wt, input_dst);
            two_pass_stats_update_rows(input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
        }
        dfb_in.pop_front(Wt);
#else
        // Keep the first two tiles in otherwise idle DEST slots and the final pass-one
        // tile in input_dst. var_dst must stay clean because finalization writes only
        // the result row. Replay retained tiles in order, using the now-free retained
        // slot for any middle tiles.
        constexpr uint32_t num_front_retained_limit = 2;
        constexpr uint32_t num_front_retained = Wt < num_front_retained_limit ? Wt : num_front_retained_limit;
        for (uint32_t wt = 0; wt < num_front_retained; ++wt) {
            const uint32_t stats_input_dst = wt == 0 ? retained_input_dst : wt;
            two_pass_stats_update_rows(stats_input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
        }
        if constexpr (Wt > num_front_retained) {
            for (uint32_t wt = num_front_retained; wt < Wt - 1; ++wt) {
                dfb_in.wait_front(onetile);
                transpose_tile(dfb::in, 0, retained_input_dst);
                dfb_in.pop_front(onetile);
                two_pass_stats_update_rows(retained_input_dst, 0, tile_width);
            }
            two_pass_stats_update_rows(input_dst, 0, last_tile_rows);
        }
#endif
        two_pass_stats_finalize_to_row(mean_dst, two_pass_variance_reciprocal);
        tile_regs_commit();

        // Pack variance and transpose back to column format
        dfb_var.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::var);
        pack_tile(var_dst, dfb::var);
        tile_regs_release();
        dfb_var.push_back(onetile);

        dfb_var.wait_front(onetile);
        reconfig_data_format_srca(dfb::var);
        transpose_init(dfb::var);
        tile_regs_acquire();
        transpose_tile(dfb::var, 0, var_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
#ifdef WELFORD_POST_MUL
        // Apply the user scalar to the reduced output: var(s*x)=s^2 var(x), std(s*x)=|s| std(x).
        // mul_unary_tile is an SFPU op operating on DEST at full fp32 precision.
        binop_with_scalar_tile_init();
        mul_unary_tile(var_dst, post_mul_scaler_bits);
#endif
        tile_regs_commit();
        dfb_var.pop_front(onetile);

        // Pack transposed variance to output
        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out);
        pack_tile(var_dst, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);

    }  // NCHt loop
}
