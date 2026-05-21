// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime args:
    // Total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    auto NCHt = get_arg(args::work_units_per_core);

    // Compile-time args:
    // Number of tiles along the W (reduction) dimension.
    constexpr auto Wt = get_arg(args::Wt);
    // The actual number of elements along W (before tiling).
    constexpr auto W = get_arg(args::W);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr auto tile_width = get_arg(args::tile_width);
    // Whether input scaling is required.
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_arg(args::correction) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in_obj(dfb::in_dfb);
    DataflowBuffer cb_scalar_obj(dfb::scaler_dfb);
    DataflowBuffer cb_out_obj(dfb::out_dfb);
    DataflowBuffer cb_var_obj(dfb::var_dfb);
    // cb_scaled is bound unconditionally on the host (per the patterns catalog
    // Conditional / optional DFB bindings entry). When do_scale=false the
    // wrapper sits unused; the if constexpr (do_scale) block below elides all
    // of its uses.
    DataflowBuffer cb_scaled_obj(dfb::scaled_dfb);

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

    compute_kernel_hw_startup(dfb::in_dfb, dfb::out_dfb);
    pack_reconfig_data_format(dfb::out_dfb);

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
            // Explicit srca reconfig is required because the output packing
            // phase (below) calls reconfig_data_format_srca(cb_var) which
            // changes the format on the 2nd+ NCHt iterations.
            reconfig_data_format_srca(dfb::in_dfb);
            transpose_wh_init_short(dfb::in_dfb);
            tile_regs_acquire();
        }

        // Welford SFPU state (running mean in LREG4, M2 in LREG5)
        // persists across DST cycles because LREGs are separate from
        // the DST register file managed by tile_regs_acquire/release.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if constexpr (do_scale) {
                // --- Scale step: multiply input tile by scalar ---
                cb_in_obj.wait_front(onetile);
                tile_regs_acquire();
                // Explicit srca reconfig is required because the output packing
                // phase (below) calls reconfig_data_format_srca(cb_var) which
                // sets the unpacker to cb_var's format (Float32 when fp32 dest
                // accumulation is enabled).  mul_tiles_bcast_scalar_init_short
                // does not fully reconfigure the data format, so without this
                // call the unpacker would read cb_in (Float16_b) data using the
                // stale Float32 format, producing garbage on the 2nd+ NCHt
                // iterations.
                reconfig_data_format_srca(dfb::in_dfb);
                mul_tiles_bcast_scalar_init_short(dfb::in_dfb, dfb::scaler_dfb);
                mul_tiles_bcast_scalar(dfb::in_dfb, dfb::scaler_dfb, 0, 0, input_dst);
                tile_regs_commit();
                cb_in_obj.pop_front(1);
                cb_scaled_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_reconfig_data_format(dfb::scaled_dfb);
                pack_tile(input_dst, dfb::scaled_dfb);
                tile_regs_release();
                cb_scaled_obj.push_back(onetile);

                // --- Transpose scaled tile back into DST ---
                cb_scaled_obj.wait_front(onetile);
                tile_regs_acquire();
                reconfig_data_format_srca(dfb::scaled_dfb);
                transpose_wh_init_short(dfb::scaled_dfb);
                transpose_wh_tile(dfb::scaled_dfb, 0, input_dst);
                cb_scaled_obj.pop_front(onetile);
            } else {
                cb_in_obj.wait_front(onetile);
                transpose_wh_tile(dfb::in_dfb, 0, input_dst);
                cb_in_obj.pop_front(onetile);
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
        pack_reconfig_data_format(dfb::var_dfb);
        pack_tile(var_dst, dfb::var_dfb);
        tile_regs_release();
        cb_var_obj.push_back(onetile);

        cb_var_obj.wait_front(onetile);
        reconfig_data_format_srca(dfb::var_dfb);
        transpose_wh_init_short(dfb::var_dfb);
        tile_regs_acquire();
        transpose_wh_tile(dfb::var_dfb, 0, var_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
        tile_regs_commit();
        cb_var_obj.pop_front(onetile);

        // Pack transposed variance to output
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out_dfb);
        pack_tile(var_dst, dfb::out_dfb);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

    }  // NCHt loop
}
