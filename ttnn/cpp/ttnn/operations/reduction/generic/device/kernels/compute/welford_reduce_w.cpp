// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"

#include "api/dataflow/circular_buffer.h"

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
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_compile_time_arg_val(5) != 0;

    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    // For FP32 input c_0 is flagged UnpackToDestFp32 by the program factory so the welford SFPU
    // intake (transpose_wh_tile) reads with full FP32 precision. For BF16 input c_0 is Default.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // True when input is FP32; gates the transpose_wh re-init / welford PreserveStats recovery
    // in the Wt-inner loop (transpose_wh_tile's UnpackToDestFp32 path clobbers the welford SFPU
    // replay buffer). On BF16 input that path is inactive, so the recovery is gated out.
    constexpr bool welford_fp32_input = get_named_compile_time_arg_val("welford_fp32_input") != 0;
    // Circular buffer where the final variance output tile is written
    // for the writer kernel to consume.
    constexpr auto cb_out = tt::CBIndex::c_16;
    // Scratch circular buffer used to hold the variance tile between
    // the two transpose steps (Welford produces row-oriented results;
    // we transpose back to column orientation via this buffer,
    // and transpose operation can't take data from the DST register).
    constexpr auto cb_var = tt::CBIndex::c_19;

    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_var_obj(cb_var);

    // Destination register indices inside the Tensix DST register file.
    // Welford's LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current transposed input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed.
    constexpr uint32_t last_tile_rows = ((W % tile_width) == 0) ? tile_width : (W % tile_width);

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm.
        // Input tiles are transposed directly from cb_in and fed to welford_update.
        // The Welford SFPU state (running mean in LREG4, M2 in LREG5) persists
        // across tile_regs_release/acquire cycles because LREGs are SFPU registers,
        // separate from the DST register file managed by tile_regs_acquire/release.

        // start_N is the cumulative sample count across tiles processed so far;
        // passed to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each sample's running-mean update.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        // transpose and welford (both SFPU-compatible) share a single DST window for the entire
        // loop: one acquire before the loop, one commit after the last tile.
        reconfig_data_format_srca(cb_in);
        // cb_in's UnpackToDestFp32 mode (FP32 input only) was already programmed by
        // compute_kernel_hw_startup(cb_in, cb_out) at kernel entry, so _init_short is
        // enough here. For BF16 input cb_in is Default mode and the same call works.
        transpose_wh_init_short(cb_in);
        tile_regs_acquire();

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_in_obj.wait_front(onetile);
            if constexpr (welford_fp32_input) {
                // Re-records the transpose-dest setup at math-thread replay slots [16, 32).
                transpose_wh_init_short(cb_in);
            }
            transpose_wh_tile(cb_in, 0, input_dst);
            cb_in_obj.pop_front(onetile);

            // For fp32 input, transpose_wh_tile takes the UnpackToDest path whose math-side init
            // overwrites the upper half of the SFPU replay buffer (slots [16, 32)), clobbering
            // welford's recurrence. welford_init<WelfordInitMode::PreserveStats>() re-records all 32 slots without
            // clearing LREG4/5 (which would lose the running mean/M2 accumulator). UNPACK A is left in transpose=1 by
            // transpose_wh_tile; welford_update is pure SFPU and does not consume that state, and the next iteration's
            // transpose_wh_init[_short] reprograms it.
            //
            // For bf16 input the unpack-to-DEST fp32 path is inactive: transpose_wh_tile routes
            // through SrcA without touching the SFPU replay buffer, so the recovery is gated out.
            if constexpr (welford_fp32_input) {
                welford_init<WelfordInitMode::PreserveStats>();
            }

            if (wt < (Wt - 1)) {
                welford_update<0>(input_dst, start_N, {});
            } else {
                // Last tile: finalize and keep DST acquired for variance packing
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance conversion:
                //   correction=false: scale_idx = W-1, reciprocal = 1/W  (population variance)
                //   correction=true:  scale_idx = W-2, reciprocal = 1/(W-1) (sample variance)
                constexpr uint32_t scale_idx = correction ? (W - 2) : (W - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                tile_regs_commit();
            }
            start_N += tile_width;
        }

        // Pack variance and transpose back to column format
        cb_var_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_var);
        pack_tile(var_dst, cb_var);
        tile_regs_release();
        cb_var_obj.push_back(onetile);

        cb_var_obj.wait_front(onetile);
        reconfig_data_format_srca(cb_var);
        transpose_wh_init_short(cb_var);
        tile_regs_acquire();
        transpose_wh_tile(cb_var, 0, var_dst);
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
        cb_var_obj.pop_front(onetile);

        // Pack transposed variance to output
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(var_dst, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

    }  // NCHt loop
}
