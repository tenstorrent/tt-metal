// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 Welford H-dimension reduction compute kernel.
// Reduces along H (rows) directly using the Welford LLK, which natively reduces
// rows and maintains per-column accumulators.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/dataflow_buffer.h"

void kernel_main() {
    // Runtime arg: number of independent column-reductions this core must perform.
    // Each column-reduction processes Ht tiles vertically and produces one output tile.
    const uint32_t NCWt = get_arg(args::NCWt);

    // Compile-time args.
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    experimental::DataflowBuffer dfb_input(dfb::input);
    experimental::DataflowBuffer dfb_scaler(dfb::scaler);
    experimental::DataflowBuffer dfb_output(dfb::output);

    const uint32_t cb_in = dfb_input.get_id();
    const uint32_t cb_scalar = dfb_scaler.get_id();
    const uint32_t cb_out = dfb_output.get_id();

    // DST register layout: input_dst (0) — scratch for current input tile,
    // mean_dst (1) — running / final mean accumulator,
    // var_dst (2) — running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // Number of valid rows in the last tile in height dimension.
    // Welford's LLK processes rows naturally, so we skip padding rows in the last
    // tile via welford_update_rows.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);

    if constexpr (do_scale) {
        dfb_scaler.wait_front(onetile);  // scalar tile stays resident across all columns
    }

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        // start_N is the cumulative row count across tiles processed so far; passed
        // to the Welford LLK so it can compute the correct 1/(N+1) reciprocal for
        // each row's running-mean update.
        uint32_t start_N = 0;
        welford_init();

        // Process one tile-column along H while keeping a single running Welford state.
        // Welford's running accumulators (mean in LREG4, M2 in LREG5) live in SFPU
        // local registers (LREGs), which are separate from the DST register file.
        // The Welford state survives across tile_regs_release/acquire cycles — only
        // DST contents are affected by the handshake, not the SFPU accumulators.
        //
        // do_scale path mixes two incompatible operation types each iteration:
        //   1. mul_tiles_bcast_scalar  — an FPU operation
        //   2. welford_update          — an SFPU operation
        // We do a full acquire/commit/wait/release on non-final iterations even
        // though nothing is packed; the handshake leaves the scalar-mul configuration
        // in a clean state before the next iteration reconfigures the compute engines.
        // In the !do_scale path, only SFPU-compatible ops run, so a single DST window
        // suffices for the whole loop.
        if constexpr (!do_scale) {
            copy_tile_to_dst_init_short(cb_in);
            tile_regs_acquire();
        }

        for (uint32_t ht = 0; ht < Ht; ++ht) {
            dfb_input.wait_front(onetile);

            if constexpr (do_scale) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(cb_in, cb_scalar);
                mul_tiles_bcast_scalar(cb_in, cb_scalar, 0, 0, input_dst);

                // Reconfigure compute setup from scalar-multiply mode back to the SFPU
                // state Welford expects.
                welford_reinit(cb_in);
            } else {
                copy_tile(cb_in, 0, input_dst);
            }
            dfb_input.pop_front(onetile);

            if (ht < (Ht - 1)) {
                welford_update<0>(input_dst, start_N, {});
                if constexpr (do_scale) {
                    // Even on non-final iterations we must complete the full DST handshake
                    // here. We are not producing a packed output tile yet; this commit/wait/
                    // release fully closes out the current DST and compute-engine state after
                    // mul_tiles_bcast_scalar before the next iteration reconfigures UNPACK+MATH.
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
            } else {
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // scale_idx controls the divisor for M2 -> variance:
                //   correction=false: scale_idx = H-1, reciprocal = 1/H   (population)
                //   correction=true:  scale_idx = H-2, reciprocal = 1/(H-1) (sample)
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

        // Pack variance/std directly to output — no transpose needed for H reduction
        // because Welford natively produces results in row orientation which matches
        // the desired output layout (one row of results per column of input).
        dfb_output.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(var_dst, cb_out);
        tile_regs_release();
        dfb_output.push_back(onetile);
    }
}
