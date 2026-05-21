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

void kernel_main() {
    // Runtime arg: number of independent column-reductions this core must perform.
    // Each column-reduction processes Ht tiles vertically and produces one output tile.
    auto NCWt = get_arg(args::work_units_per_core);

    // Compile-time args:
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto H = get_arg(args::H);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in_obj(dfb::in_dfb);
    DataflowBuffer cb_scalar_obj(dfb::scaler_dfb);
    DataflowBuffer cb_out_obj(dfb::out_dfb);

    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid rows in the last tile in height dimension.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb::in_dfb, dfb::out_dfb);
    pack_reconfig_data_format(dfb::out_dfb);

    if constexpr (do_scale) {
        // Scalar tile stays resident across all columns
        cb_scalar_obj.wait_front(onetile);
    }

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        // Welford accumulation along the H dimension for one column of tiles.

        // start_N is the cumulative row count across tiles processed so far; passed
        // to the Welford LLK so it can compute the correct 1/(N+1) reciprocal
        // for each row's running-mean update.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        // Process one tile-column along H while keeping a single running Welford state.
        // (Full rationale of DST/tile_regs flow, init sequencing, and SFPU/FPU
        // interleaving lives in the legacy version of this kernel; preserved here
        // verbatim.)
        if constexpr (!do_scale) {
            copy_tile_to_dst_init_short(dfb::in_dfb);
            tile_regs_acquire();
        }

        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_in_obj.wait_front(onetile);

            if constexpr (do_scale) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(dfb::in_dfb, dfb::scaler_dfb);
                mul_tiles_bcast_scalar(dfb::in_dfb, dfb::scaler_dfb, 0, 0, input_dst);

                // Reconfigure the compute setup from scalar-multiply mode back to the
                // SFPU state that Welford expects.
                welford_reinit(dfb::in_dfb);
            } else {
                copy_tile(dfb::in_dfb, 0, input_dst);
            }
            cb_in_obj.pop_front(onetile);

            if (ht < (Ht - 1)) {
                welford_update<0>(input_dst, start_N, {});
                if constexpr (do_scale) {
                    // Even on non-final iterations we must complete the full DST handshake here.
                    // We are not producing a packed output tile yet; this commit/wait/release is
                    // only to fully close out the current DST and compute-engine state after
                    // mul_tiles_bcast_scalar, before the next iteration reconfigures UNPACK+MATH
                    // for another scaled input. Without this handshake, the next iteration can
                    // inherit stale DST or UNPACK+MATH configuration.
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
            } else {
                // Last tile: process only valid rows, then finalize
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance conversion:
                //   correction=false: scale_idx = H-1, reciprocal = 1/H  (population variance)
                //   correction=true:  scale_idx = H-2, reciprocal = 1/(H-1) (sample variance)
                constexpr uint32_t scale_idx = correction ? (H - 2) : (H - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                if constexpr (is_std) {
                    sqrt_tile_init();
                    sqrt_tile(var_dst);
                }
                tile_regs_commit();
            }
            start_N += tile_height;
        }

        // Pack variance/std directly to output -- no transpose needed for H reduction
        // because Welford natively produces results in row orientation which matches
        // the desired output layout (one row of results per column of input).
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out_dfb);
        pack_tile(var_dst, dfb::out_dfb);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
