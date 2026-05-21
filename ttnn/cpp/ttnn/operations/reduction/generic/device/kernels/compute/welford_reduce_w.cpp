// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford W-reduce compute kernel, ported to Metal 2.0.
//
// Host bindings expected (per the Welford factory's W KernelSpec):
//   compile_time_arg_bindings:
//     { {"Wt", ...}, {"W", ...}, {"tile_width", ...},
//       {"do_scale", 0|1}, {"correction", 0|1}, {"is_std", 0|1} }
//   runtime_arguments_schema.named_runtime_args: { "NCHt" }
//   dfb_bindings:
//     { INPUT (CONSUMER, name="input"),
//       SCALER (CONSUMER, name="scaler"),  (only when do_scale via cb_scalar)
//       OUTPUT (PRODUCER, name="output"),
//       SCRATCH (PRODUCER+CONSUMER self-loop, name="scratch"),   // cb_var (c_19)
//       SCALED (PRODUCER+CONSUMER self-loop, name="scaled") }    // cb_scaled (c_20),
//                                                                  only when do_scale

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
    auto NCHt = get_arg(args::NCHt);

    // Compile-time args:
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    // do_scale is gated by the DO_SCALE define (host-side, in welford_reduce factory).
    // Using #ifdef rather than `if constexpr` means the dfb::scaled accessor is only
    // referenced when the host actually binds the SCALED DFB; otherwise the name need
    // not exist in kernel_args_generated.h.
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in_obj(dfb::input);
    DataflowBuffer cb_out_obj(dfb::output);
    DataflowBuffer cb_var_obj(dfb::scratch);
    // Scaler DFB is bound on the compute kernel as CONSUMER unconditionally.
    // The reader unconditionally produces a scaler tile (via prepare_reduce_scaler);
    // when do_scale=false we still pop it so the DFB has a balanced producer/consumer.
    DataflowBuffer cb_scalar_obj(dfb::scaler);

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

    compute_kernel_hw_startup(dfb::input, dfb::output);
    pack_reconfig_data_format(dfb::output);

    // Scalar tile stays resident across all rows (waited unconditionally so the scaler DFB
    // has a producer/consumer pair; the value is only used when do_scale).
    cb_scalar_obj.wait_front(onetile);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm.
        // start_N is the cumulative sample count across tiles processed so far.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

#ifndef DO_SCALE
        // Explicit srca reconfig is required because the output packing
        // phase (below) calls reconfig_data_format_srca(dfb::scratch) which
        // changes the format on the 2nd+ NCHt iterations.
        reconfig_data_format_srca(dfb::input);
        transpose_wh_init_short(dfb::input);
        tile_regs_acquire();
#endif

        for (uint32_t wt = 0; wt < Wt; ++wt) {
#ifdef DO_SCALE
            // --- Scale step: multiply input tile by scalar ---
            cb_in_obj.wait_front(onetile);
            tile_regs_acquire();
            reconfig_data_format_srca(dfb::input);
            mul_tiles_bcast_scalar_init_short(dfb::input, dfb::scaler);
            mul_tiles_bcast_scalar(dfb::input, dfb::scaler, 0, 0, input_dst);
            tile_regs_commit();
            cb_in_obj.pop_front(1);
            DataflowBuffer cb_scaled_obj(dfb::scaled);
            cb_scaled_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_reconfig_data_format(dfb::scaled);
            pack_tile(input_dst, dfb::scaled);
            tile_regs_release();
            cb_scaled_obj.push_back(onetile);

            // --- Transpose scaled tile back into DST ---
            cb_scaled_obj.wait_front(onetile);
            tile_regs_acquire();
            reconfig_data_format_srca(dfb::scaled);
            transpose_wh_init_short(dfb::scaled);
            transpose_wh_tile(dfb::scaled, 0, input_dst);
            cb_scaled_obj.pop_front(onetile);
#else
            cb_in_obj.wait_front(onetile);
            transpose_wh_tile(dfb::input, 0, input_dst);
            cb_in_obj.pop_front(onetile);
#endif

            if (wt < (Wt - 1)) {
                welford_update<0>(input_dst, start_N, {});
#ifdef DO_SCALE
                tile_regs_commit();
                tile_regs_wait();
                tile_regs_release();
#endif
            } else {
                // Last tile: finalize and keep DST acquired for variance packing
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance conversion.
                constexpr uint32_t scale_idx = correction ? (W - 2) : (W - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                tile_regs_commit();
            }
            start_N += tile_width;
        }

        // Pack variance and transpose back to column format
        cb_var_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::scratch);
        pack_tile(var_dst, dfb::scratch);
        tile_regs_release();
        cb_var_obj.push_back(onetile);

        cb_var_obj.wait_front(onetile);
        reconfig_data_format_srca(dfb::scratch);
        transpose_wh_init_short(dfb::scratch);
        tile_regs_acquire();
        transpose_wh_tile(dfb::scratch, 0, var_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
        tile_regs_commit();
        cb_var_obj.pop_front(onetile);

        // Pack transposed variance to output
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::output);
        pack_tile(var_dst, dfb::output);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

    }  // NCHt loop
}
