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

#ifdef WELFORD_POST_MUL
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Runtime args:
    // Total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    uint32_t NCHt = get_arg_val<uint32_t>(0);

    // Compile-time args:
    // Number of tiles along the W (reduction) dimension.
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // The actual number of elements along W (before tiling).
    constexpr uint32_t W = get_compile_time_arg_val(1);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr uint32_t tile_width = get_compile_time_arg_val(2);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
#endif
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_compile_time_arg_val(4) != 0;

    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    // For FP32 input c_0 is flagged UnpackToDestFp32 by the program factory so the statistics SFPU
    // intake (transpose_tile) reads with full FP32 precision. For BF16 input c_0 is Default.
    constexpr auto dfb_in = tt::CBIndex::c_0;
    constexpr uint32_t two_pass_mean_reciprocal = get_named_compile_time_arg_val("two_pass_mean_reciprocal");
    constexpr uint32_t two_pass_variance_reciprocal =
        get_named_compile_time_arg_val("two_pass_variance_reciprocal");
    // Circular buffer where the final variance output tile is written
    // for the writer kernel to consume.
    constexpr auto dfb_out = tt::CBIndex::c_16;
    // Scratch circular buffer used to hold the variance tile between
    // the two transpose steps (Welford produces row-oriented results;
    // we transpose back to column orientation via this buffer,
    // and transpose operation can't take data from the DST register).
    constexpr auto dfb_var = tt::CBIndex::c_19;

    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_out_obj(dfb_out);
    DataflowBuffer dfb_var_obj(dfb_var);

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

    compute_kernel_hw_startup(dfb_in, dfb_out);
    pack_reconfig_data_format(dfb_out);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        reconfig_data_format_srca(dfb_in);
        transpose_init(dfb_in);
        tile_regs_acquire();
        two_pass_stats_init_shifted();

        for (uint32_t wt = 0; wt < Wt; ++wt) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
            // Do not pop pass-one tiles: the enlarged CB holds the complete
            // reduction row so pass two can index the same L1 pages directly.
            dfb_in_obj.wait_front(wt + 1);
            transpose_tile(dfb_in, wt, input_dst);
            constexpr uint32_t stats_input_dst = input_dst;
#else
            dfb_in_obj.wait_front(onetile);
#ifdef WELFORD_TWO_PASS_BFP8_INPUT
            // Keep var_dst clean: finalization writes only the variance row,
            // and stale large values elsewhere would coarsen BFP8 block packing.
            const uint32_t stats_input_dst = wt < 2 ? (wt == 0 ? retained_input_dst : mean_dst) : input_dst;
#else
            const uint32_t stats_input_dst = wt < 3 ? (wt == 0 ? retained_input_dst : wt) : input_dst;
#endif
            transpose_tile(dfb_in, 0, stats_input_dst);
            dfb_in_obj.pop_front(onetile);
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
            transpose_tile(dfb_in, wt, input_dst);
            two_pass_stats_update_rows<true>(input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
        }
        dfb_in_obj.pop_front(Wt);
#else
        // Keep the first two or three tiles in otherwise idle DEST slots and the final
        // pass-one tile in input_dst. BFP8 retains one fewer tile so var_dst
        // stays clean for sparse result packing. Replay retained tiles in
        // order, using the now-free retained slot for any middle tiles.
#ifdef WELFORD_TWO_PASS_BFP8_INPUT
        constexpr uint32_t num_front_retained_limit = 2;
#else
        constexpr uint32_t num_front_retained_limit = 3;
#endif
        constexpr uint32_t num_front_retained = Wt < num_front_retained_limit ? Wt : num_front_retained_limit;
        for (uint32_t wt = 0; wt < num_front_retained; ++wt) {
            const uint32_t stats_input_dst = wt == 0 ? retained_input_dst : wt;
            two_pass_stats_update_rows<true>(stats_input_dst, 0, wt == Wt - 1 ? last_tile_rows : tile_width);
        }
        if constexpr (Wt > num_front_retained) {
            for (uint32_t wt = num_front_retained; wt < Wt - 1; ++wt) {
                dfb_in_obj.wait_front(onetile);
                transpose_tile(dfb_in, 0, retained_input_dst);
                dfb_in_obj.pop_front(onetile);
                two_pass_stats_update_rows<true>(retained_input_dst, 0, tile_width);
            }
            two_pass_stats_update_rows<true>(input_dst, 0, last_tile_rows);
        }
#endif
        two_pass_stats_finalize_to_row(mean_dst, two_pass_variance_reciprocal);
        tile_regs_commit();

        // Pack variance and transpose back to column format
        dfb_var_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_var);
        pack_tile(var_dst, dfb_var);
        tile_regs_release();
        dfb_var_obj.push_back(onetile);

        dfb_var_obj.wait_front(onetile);
        reconfig_data_format_srca(dfb_var);
        transpose_init(dfb_var);
        tile_regs_acquire();
        transpose_tile(dfb_var, 0, var_dst);
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
        dfb_var_obj.pop_front(onetile);

        // Pack transposed variance to output
        dfb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_out);
        pack_tile(var_dst, dfb_out);
        tile_regs_release();
        dfb_out_obj.push_back(onetile);

    }  // NCHt loop
}
