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
    // Whether input scaling is required.
    constexpr bool do_scale = get_compile_time_arg_val(3) != 0;
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_compile_time_arg_val(5) != 0;

    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    // For FP32 input + do_scale=false: c_0 is flagged UnpackToDestFp32 by the program factory
    // so the welford SFPU intake (transpose_wh_tile) reads with full FP32 precision.
    // For FP32 input + do_scale=true: c_0 stays Default for the FPU mul SrcA read; precision
    // preservation lives on cb_scaled (c_20) below.
    // For BF16 input: c_0 is Default regardless.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // True when input is FP32; gates full hw_configure pairs in the do_scale Wt-inner loop
    // to flip UNPACK between cb_in (Default, for FPU mul) and cb_scaled (UnpackToDestFp32,
    // for welford-intake transpose). On BF16 input neither CB carries UnpackToDestFp32, so
    // _init_short calls suffice.
    constexpr bool welford_fp32_input = get_named_compile_time_arg_val("welford_fp32_input") != 0;
    // Scalar tile produced by the reader via generate_reduce_scaler.
    // Used to scale every input tile before Welford processing.
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    // Circular buffer where the final variance output tile is written
    // for the writer kernel to consume.
    constexpr auto cb_out = tt::CBIndex::c_16;
    // Scratch circular buffer used to hold the variance tile between
    // the two transpose steps (Welford produces row-oriented results;
    // we transpose back to column orientation via this buffer,
    // and transpose operation can't take data from the DST register).
    constexpr auto cb_var = tt::CBIndex::c_19;
    // 1-tile intermediate: holds the scaled input tile between the
    // mul_tiles_bcast_scalar (scale step) and transpose_wh_tile (Welford step).
    // Only used when do_scale is true.
    // The reason this is needed is because mul_tiles_bcast_scalar writes data
    // to the DST register, but transpose_wh_tile is an unpack operation, so
    // it expects data in a CB. Thus, this CB is used to hold the scaled input tile.
    constexpr auto cb_scaled = tt::CBIndex::c_20;

    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scalar_obj(cb_scalar);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_var_obj(cb_var);
    CircularBuffer cb_scaled_obj(cb_scaled);

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

    if constexpr (do_scale) {
        // Scalar tile stays resident across all rows
        cb_scalar_obj.wait_front(onetile);
    }

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm.
        // When do_scale is true, each input tile is first multiplied by the scalar,
        // packed to cb_scaled, then transposed and fed to welford_update.
        // When do_scale is false, input tiles are transposed directly from cb_in.
        // The Welford SFPU state (running mean in LREG4, M2 in LREG5) persists
        // across tile_regs_release/acquire cycles because LREGs are SFPU registers,
        // separate from the DST register file managed by tile_regs_acquire/release.

        // start_N is the cumulative sample count across tiles processed so far;
        // passed to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each sample's running-mean update.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        // When scaling, DST must be acquired/released per tile because
        // the FPU mul is incompatible with SFPU Welford. The scaled
        // result is packed to cb_scaled so that transpose_wh_tile (an
        // unpack operation) can read it back.
        // Without scaling, transpose and welford (both SFPU-compatible)
        // can share a single DST window for the entire loop.
        // On the do_scale path, transpose_wh_init_short(cb_scaled) runs before each
        // welford_update; it re-inits UNPACK and MATH via llk_math_eltwise_unary_datacopy_init
        // (same MATH-side effect as welford_reinit), so a separate welford_reinit after the mul
        // is not required here.
        if constexpr (!do_scale) {
            reconfig_data_format_srca(cb_in);
            // cb_in's UnpackToDestFp32 mode (FP32 input only) was already programmed by
            // compute_kernel_hw_startup(cb_in, cb_out) at kernel entry, so _init_short is
            // enough here. For BF16 input cb_in is Default mode and the same call works.
            transpose_wh_init_short(cb_in);
            tile_regs_acquire();
        }
        // In contrast, on do_scale path, full init_bcast and full transpose_wh_init happen inside
        // the Wt loop (each iteration) to pair UNPACK mode flips for cb_in (Default, FPU mul SrcA)
        // and cb_scaled (UnpackToDestFp32, unpack-to-DEST consumer for the welford-intake transpose).
        // The first iteration's init_bcast also resets the leaked UnpackToDest mode from the
        // previous NCHt iteration's final transpose_wh_init(cb_var, cb_out), so no separate
        // outer-loop reset is needed.

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if constexpr (do_scale) {
                // --- Scale step: multiply input tile by scalar ---
                cb_in_obj.wait_front(onetile);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_in);
                if constexpr (welford_fp32_input) {
                    // Full init_bcast each iter: programs UNPACK hw_configure for cb_in
                    // (Default mode), resetting any UnpackToDest mode left by either the previous
                    // wt's cb_scaled transpose or the previous NCHt's cb_var final transpose.
                    // On FP32 input, the _init_short variants are not sufficient because they
                    // skip llk_unpack_hw_configure, which is the only call that reprograms the
                    // unpack-to-DEST mode bit; reconfig_data_format_srca reprograms only
                    // SrcA's src/dst data format and tile geometry, not the unpack-to-DEST bit.
                    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_in, cb_scalar, cb_scaled);
                } else {
                    mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                }
                mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);
                tile_regs_commit();
                cb_in_obj.pop_front(1);
                cb_scaled_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_reconfig_data_format(cb_scaled);
                pack_tile(input_dst, cb_scaled);
                tile_regs_release();
                cb_scaled_obj.push_back(onetile);

                // --- Transpose scaled tile back into DST ---
                cb_scaled_obj.wait_front(onetile);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_scaled);
                if constexpr (welford_fp32_input) {
                    // Full transpose_wh_init each iter: programs UNPACK hw_configure for
                    // cb_scaled (UnpackToDestFp32 mode), preserving the FPU mul output's
                    // mantissa bits past TF32. The next wt iter's init_bcast resets UNPACK
                    // back to cb_in's Default mode before the FPU mul reads SrcA again.
                    transpose_wh_init(cb_scaled, cb_var);
                } else {
                    transpose_wh_init_short(cb_scaled);
                }
                transpose_wh_tile(cb_scaled, 0, input_dst);
                cb_scaled_obj.pop_front(onetile);
            } else {
                cb_in_obj.wait_front(onetile);
                if constexpr (welford_fp32_input) {
                    // Re-records the transpose-dest setup at math-thread replay slots [16, 32)
                    // and reprograms UNPACK A for a transposed read (the previous iteration's
                    // welford_reinit toggled UNPACK A back to transpose=0).
                    transpose_wh_init_short(cb_in);
                }
                transpose_wh_tile(cb_in, 0, input_dst);
                cb_in_obj.pop_front(onetile);
            }

            // For fp32 input, transpose_wh_tile takes the UnpackToDest path which uses FPU
            // MOV ops and reprograms the SFPU replay buffer. Re-establish welford state:
            //   1. welford_reinit re-establishes UNPACK+MATH datacopy config (parallel of the
            //      do_scale path which calls this after mul_tiles_bcast_scalar)
            //   2. llk_math_welfords_sfpu_init re-programs the replay buffer with the welford
            //      recurrence, without clearing LREG4/5 (which would lose the running
            //      mean/M2 accumulator).
            //
            // For bf16 input the unpack-to-DEST fp32 path is inactive: transpose_wh_tile routes
            // through SrcA without touching the SFPU replay buffer, and welford_update is pure
            // SFPU and does not disturb the UNPACK/MATH datacopy state set by the pre-loop
            // transpose_wh_init_short, so the per-iteration init/reinit triple is gated out.
            if constexpr (welford_fp32_input) {
                welford_reinit(cb_in);
                MATH((llk_math_welfords_sfpu_init()));
            }

            if (wt < (Wt - 1)) {
                welford_update<0>(input_dst, start_N, {});
                if constexpr (do_scale) {
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
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
        if constexpr (welford_fp32_input) {
            // cb_var is UnpackToDestFp32 for FP32 input; needs full hw_configure to activate
            // the unpack-to-DEST mode.
            transpose_wh_init(cb_var, cb_out);
        } else {
            transpose_wh_init_short(cb_var);
        }
        tile_regs_acquire();
        transpose_wh_tile(cb_var, 0, var_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
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
