// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

ALWI void MUL_TILES(uint32_t in0_id, uint32_t in1_id, uint32_t out_id, uint32_t num_tiles) {
    // Multiply input by cos or sin
    DataflowBuffer dfb_in0(in0_id);
    DataflowBuffer dfb_in1(in1_id);
    DataflowBuffer dfb_out(out_id);

    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    dfb_out.reserve_back(num_tiles);

    tile_regs_acquire();
    mul_init(in0_id, in1_id);
    mul_tiles(in0_id, in1_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_id);
    tile_regs_release();
    dfb_out.push_back(num_tiles);
    dfb_in0.pop_front(num_tiles);
    dfb_in1.pop_front(num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t half_Wt = get_arg(args::half_Wt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_rotated_in(dfb::rotated_in);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_scalar(dfb::scalar);
    DataflowBuffer dfb_rotated_in_interm(dfb::rotated_in_interm);
    DataflowBuffer dfb_cos_interm(dfb::cos_interm);
    DataflowBuffer dfb_sin_interm(dfb::sin_interm);
    DataflowBuffer dfb_out(dfb::out);

    dfb_scalar.wait_front(onetile);

    compute_kernel_hw_startup(dfb::rotated_in, dfb::scalar, dfb::rotated_in_interm);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                reconfig_data_format(dfb::rotated_in, dfb::scalar);
                pack_reconfig_data_format(dfb::rotated_in_interm);
                dfb_rotated_in.wait_front(onetile);
                dfb_rotated_in_interm.reserve_back(onetile);
                tile_regs_acquire();
                mul_bcast_scalar_init(dfb::rotated_in, dfb::scalar);
                mul_tiles_bcast_scalar(dfb::rotated_in, dfb::scalar, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb::rotated_in_interm);
                tile_regs_release();
                dfb_rotated_in_interm.push_back(onetile);
                dfb_rotated_in.pop_front(onetile);
                reconfig_data_format_srcb(dfb::scalar, dfb::sin);
                pack_reconfig_data_format(dfb::rotated_in_interm, dfb::sin_interm);
                // Multiply rotated input by sin
                MUL_TILES(dfb::rotated_in_interm, dfb::sin, dfb::sin_interm, onetile);
            } else {
                reconfig_data_format(dfb::rotated_in, dfb::sin);
                pack_reconfig_data_format(dfb::out, dfb::sin_interm);
                // Multiply rotated input by sin
                MUL_TILES(dfb::rotated_in, dfb::sin, dfb::sin_interm, onetile);
            }

            // Multiply input by cos
            MUL_TILES(dfb::in, dfb::cos, dfb::cos_interm, onetile);

            // Add applied sin/cos tensors
            dfb_cos_interm.wait_front(onetile);
            dfb_sin_interm.wait_front(onetile);
            dfb_out.reserve_back(onetile);

            reconfig_data_format_srca(dfb::rotated_in, dfb::cos_interm);
            pack_reconfig_data_format(dfb::cos_interm, dfb::out);
            tile_regs_acquire();
            add_init(dfb::cos_interm, dfb::sin_interm);
            add_tiles(dfb::cos_interm, dfb::sin_interm, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();

            dfb_out.push_back(onetile);
            dfb_cos_interm.pop_front(onetile);
            dfb_sin_interm.pop_front(onetile);
        }
    }
}
