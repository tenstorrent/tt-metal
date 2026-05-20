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

    // Circular buffer that the reader kernel fills with input tiles. Declared Default-mode so
    // FPU mul_tiles_bcast_scalar's SrcA read works on do_scale=true.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // Second buffer index aliased onto cb_in for the welford SFPU consumer when FP32 precision
    // is needed. For BF16 the named alias-index slot is unused (the alias is not declared on the
    // host side), and the conditional collapses cb_in_welford_alias onto cb_in. Mirrors
    // cb_x_welford in layernorm_sharded_welford.cpp.
    constexpr auto cb_in_welford_alias_named =
        static_cast<tt::CBIndex>(get_named_compile_time_arg_val("cb_in_welford_alias"));
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    constexpr auto cb_in_welford_alias = welford_fp32_alias ? cb_in_welford_alias_named : cb_in;
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
    CircularBuffer cb_in_welford_alias_obj(cb_in_welford_alias);
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
            // Read transpose input via the welford alias (UnpackToDestFp32) to preserve FP32
            // mantissa for the welford SFPU consumer. cb_in itself stays Default-mode for the
            // FPU mul on the do_scale path; the alias shares its backing allocation.
            reconfig_data_format_srca(cb_in_welford_alias);
            if constexpr (welford_fp32_alias) {
                // Full transpose_wh hw init when the alias is active. The alias buffer index
                // isn't visible to compute_kernel_hw_startup at kernel entry (only cb_in is),
                // so we run the full init once to program UNPACK / MATH / PACK hw_configure
                // for it -- this is what makes the UnpackToDestFp32 mode flagged on the alias
                // descriptor actually take effect on the unpacker. Mirrors layernorm's
                // welford-intake pattern.
                transpose_wh_init(cb_in_welford_alias, cb_var);
            } else {
                transpose_wh_init_short(cb_in_welford_alias);
            }
            tile_regs_acquire();
        } else if constexpr (welford_fp32_alias) {
            // The final transpose_wh_tile(cb_var, ...) at the end of the previous NCHt iteration
            // calls transpose_wh_init(cb_var, cb_out) which programs UNPACK hw_configure for
            // cb_var, leaving the unpacker in UnpackToDestFp32 mode (because c_19 cb_var is
            // flagged UnpackToDestFp32). On the do_scale path, the next NCHt iteration's first
            // op is mul_tiles_bcast_scalar, whose _init_short does not call hw_configure --
            // so without this re-init the FPU mul reads SrcA via the leaked UnpackToDest path
            // and produces zeros. Run the full init_bcast once per NCHt iteration to put UNPACK
            // back into cb_in's Default mode before the inner mul loop.
            init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_in, cb_scalar, cb_scaled);
        }

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if constexpr (do_scale) {
                // --- Scale step: multiply input tile by scalar ---
                cb_in_obj.wait_front(onetile);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_in);
                mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);
                tile_regs_commit();
                cb_in_obj.pop_front(1);
                if constexpr (welford_fp32_alias) {
                    // Alias is not read on the do_scale path (FPU mul reads cb_in directly), but
                    // the reader pushed to it; pop to keep its FIFO state in lock-step with cb_in.
                    cb_in_welford_alias_obj.pop_front(onetile);
                }
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
                // cb_scaled is NOT flagged UnpackToDestFp32; _init_short here doesn't touch
                // UNPACK's unpack_to_dest mode. The TF32 readback truncates cb_scaled to ~10
                // mantissa bits -- this is a precision floor on the do_scale path that we
                // accept to avoid the within-wt-iter UNPACK-mode leak (see program factory).
                transpose_wh_init_short(cb_scaled);
                transpose_wh_tile(cb_scaled, 0, input_dst);
                cb_scaled_obj.pop_front(onetile);
            } else {
                // Wait on whichever CB is read: alias when active, cb_in otherwise.
                // The alias has its own FIFO state (rd/wr pointers); the reader push_back's the
                // alias every tile so it stays in lock-step with cb_in. Without popping the
                // alias every iteration, its rd_ptr would stay frozen and subsequent NCHt
                // iterations would re-read the same slot's stale data.
                if constexpr (welford_fp32_alias) {
                    cb_in_welford_alias_obj.wait_front(onetile);
                } else {
                    cb_in_obj.wait_front(onetile);
                }
                transpose_wh_init_short(cb_in_welford_alias);
                transpose_wh_tile(cb_in_welford_alias, 0, input_dst);
                if constexpr (welford_fp32_alias) {
                    cb_in_welford_alias_obj.pop_front(onetile);
                }
                cb_in_obj.pop_front(onetile);
            }

            // For fp32 input, transpose_wh_tile takes the UnpackToDest path which uses FPU
            // MOV ops and reprograms the SFPU replay buffer. Re-establish welford state:
            //   1. welford_reinit re-establishes UNPACK+MATH datacopy config (parallel of the
            //      do_scale path which calls this after mul_tiles_bcast_scalar)
            //   2. llk_math_welfords_sfpu_init re-programs the replay buffer with the welford
            //      recurrence, without clearing LREG4/5 (which would lose the running
            //      mean/M2 accumulator).
            welford_reinit(cb_in_welford_alias);
            MATH((llk_math_welfords_sfpu_init()));

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
        if constexpr (welford_fp32_alias) {
            // cb_var is UnpackToDestFp32 for FP32 input; needs full hw_configure to activate
            // the unpack-to-DEST mode (same rationale as the welford-intake alias init above).
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
