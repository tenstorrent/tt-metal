// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 Welford W-dimension reduction compute kernel.
//
// Migration notes:
//   - Compile-time args are bound by name (args::Wt, args::W, args::tile_width,
//     args::do_scale, args::correction, args::is_std).
//   - Runtime arg NCHt is bound by name.
//   - DataflowBuffers are bound by name. cb_var is always bound; cb_scaled is
//     always bound but only used when do_scale is true (the if constexpr block
//     gates its references; the wrapper itself just stores the buffer id).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    // Runtime arg: total outer-loop iterations (N * C * Ht).
    const uint32_t NCHt = get_arg(args::NCHt);

    // Compile-time args.
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    experimental::DataflowBuffer cb_in_obj(dfb::input);
    experimental::DataflowBuffer cb_scalar_obj(dfb::scaler);
    experimental::DataflowBuffer cb_out_obj(dfb::output);
    experimental::DataflowBuffer cb_var_obj(dfb::var);
    experimental::DataflowBuffer cb_scaled_obj(dfb::scaled);

    const uint32_t cb_in = cb_in_obj.get_id();
    const uint32_t cb_scalar = cb_scalar_obj.get_id();
    const uint32_t cb_out = cb_out_obj.get_id();
    const uint32_t cb_var = cb_var_obj.get_id();
    const uint32_t cb_scaled = cb_scaled_obj.get_id();

    // Welford's LLK uses three adjacent DST registers:
    //   input_dst (0) — scratch for the current transposed input tile,
    //   mean_dst  (1) — running / final mean accumulator,
    //   var_dst   (2) — running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // Number of valid columns in the last tile in width dimension.
    // Welford's LLK is given transposed data, so we skip rows when we want to
    // skip columns in the original (untransposed) tile.
    constexpr uint32_t last_tile_rows = ((W % tile_width) == 0) ? tile_width : (W % tile_width);

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);

    if constexpr (do_scale) {
        cb_scalar_obj.wait_front(onetile);  // scalar tile stays resident across all rows
    }

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // start_N is the cumulative sample count across tiles processed so far;
        // passed to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each sample's running-mean update.
        uint32_t start_N = 0;
        welford_init();

        if constexpr (!do_scale) {
            // Explicit srca reconfig is required because the output packing phase (below) calls
            // reconfig_data_format_srca(cb_var) which changes the format on the 2nd+ NCHt iterations.
            reconfig_data_format_srca(cb_in);
            transpose_wh_init_short(cb_in);
            tile_regs_acquire();
        }

        // Welford SFPU state (running mean in LREG4, M2 in LREG5) persists across
        // DST cycles because LREGs are separate from the DST register file.
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if constexpr (do_scale) {
                // Scale step: multiply input tile by scalar.
                cb_in_obj.wait_front(onetile);
                tile_regs_acquire();
                // Explicit srca reconfig: pack phase below calls reconfig_data_format_srca(cb_var)
                // which sets the unpacker to cb_var's format (Float32 when fp32 dest acc enabled).
                // mul_tiles_bcast_scalar_init_short does not fully reconfigure the data format, so
                // without this the unpacker would read cb_in (Float16_b) using the stale Float32
                // format and produce garbage on 2nd+ NCHt iterations.
                reconfig_data_format_srca(cb_in);
                mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);
                tile_regs_commit();
                cb_in_obj.pop_front(1);
                cb_scaled_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_reconfig_data_format(cb_scaled);
                pack_tile(input_dst, cb_scaled);
                tile_regs_release();
                cb_scaled_obj.push_back(onetile);

                // Transpose scaled tile back into DST.
                cb_scaled_obj.wait_front(onetile);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_scaled);
                transpose_wh_init_short(cb_scaled);
                transpose_wh_tile(cb_scaled, 0, input_dst);
                cb_scaled_obj.pop_front(onetile);
            } else {
                cb_in_obj.wait_front(onetile);
                transpose_wh_tile(cb_in, 0, input_dst);
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
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance:
                //   correction=false: scale_idx = W-1, reciprocal = 1/W   (population)
                //   correction=true:  scale_idx = W-2, reciprocal = 1/(W-1) (sample)
                constexpr uint32_t scale_idx = correction ? (W - 2) : (W - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                tile_regs_commit();
            }
            start_N += tile_width;
        }

        // Pack variance and transpose back to column format.
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
        tile_regs_commit();
        cb_var_obj.pop_front(onetile);

        // Pack transposed variance to output.
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(var_dst, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
