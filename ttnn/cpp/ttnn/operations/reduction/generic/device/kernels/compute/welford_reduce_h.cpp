// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford H-dimension reduction kernel, ported to Metal 2.0.
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
    auto NCWt = get_arg(args::NCWt);

    // Compile-time args:
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr bool do_scale = get_arg(args::do_scale) != 0;
    constexpr bool correction = get_arg(args::correction) != 0;
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in_obj(dfb::input);
    DataflowBuffer cb_out_obj(dfb::output);

    // Destination register indices inside the Tensix DST register file.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid rows in the last tile in height dimension.
    constexpr uint32_t last_tile_rows = ((H % tile_height) == 0) ? tile_height : (H % tile_height);

    compute_kernel_hw_startup(dfb::input, dfb::output);
    pack_reconfig_data_format(dfb::output);

    if constexpr (do_scale) {
        // Scalar tile stays resident across all columns
        DataflowBuffer cb_scalar_obj(dfb::scaler);
        cb_scalar_obj.wait_front(onetile);
    }

    for (uint32_t ncwt = 0; ncwt < NCWt; ncwt++) {
        // Welford accumulation along the H dimension for one column of tiles.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        if constexpr (!do_scale) {
            copy_tile_to_dst_init_short(dfb::input);
            tile_regs_acquire();
        }

        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_in_obj.wait_front(onetile);

            if constexpr (do_scale) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(dfb::input, dfb::scaler);
                mul_tiles_bcast_scalar(dfb::input, dfb::scaler, 0, 0, input_dst);

                // Reconfigure the compute setup from scalar-multiply mode back to the
                // SFPU state that Welford expects.
                welford_reinit(dfb::input);
            } else {
                copy_tile(dfb::input, 0, input_dst);
            }
            cb_in_obj.pop_front(onetile);

            if (ht < (Ht - 1)) {
                welford_update<0>(input_dst, start_N, {});
                if constexpr (do_scale) {
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
            } else {
                // Last tile: process only valid rows, then finalize
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
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
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::output);
        pack_tile(var_dst, dfb::output);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
