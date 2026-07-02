// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#define APPROX false
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"
#include "../accumulation_common.hpp"

void kernel_main() {
    constexpr uint32_t default_acc_value = get_arg(args::default_acc_value);

    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t tiles_per_row = get_arg(args::tiles_per_row);

    DataflowBuffer dfb_in_obj(dfb::src);
    DataflowBuffer dfb_out_obj(dfb::dst);
    DataflowBuffer dfb_acc_obj(dfb::acc);  // note: only used in compute kernel

    unary_op_init_common(dfb::src, dfb::dst);

    BINARY_OP_INIT();

    constexpr uint32_t DST_IN = 0;
    constexpr uint32_t DST_ACC = 1;

    dfb_acc_obj.reserve_back(ONE_TILE);
    dfb_acc_obj.push_back(ONE_TILE);

    for (uint32_t i = 0; i < num_rows; i++) {
        // Synchronize unpacker-packer between iterations
        // This is necessary to avoid data-races on cb_acc
        dfb_acc_obj.wait_front(ONE_TILE);
        dfb_acc_obj.pop_front(ONE_TILE);

        tile_regs_acquire();
        reconfig_data_format(dfb::acc, dfb::acc);

        fill_tile_init();
        FILL_TILE(DST_ACC, default_acc_value);

        tile_regs_commit();

        tile_regs_wait();

        // out_of_order_output to keep packing to cb_acc at the same location
        dfb_acc_obj.reserve_back(ONE_TILE);

        pack_reconfig_data_format(dfb::acc);
        pack_tile(DST_ACC, dfb::acc);
        tile_regs_release();

        dfb_acc_obj.push_back(ONE_TILE);

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            // Synchronize unpacker-packer between iterations
            dfb_acc_obj.wait_front(ONE_TILE);

            tile_regs_acquire();
            dfb_in_obj.wait_front(ONE_TILE);

            reconfig_data_format(dfb::src, dfb::src);
            copy_tile_to_dst_init_short(dfb::src);
            copy_tile(dfb::src, 0, DST_IN);

            reconfig_data_format(dfb::acc, dfb::acc);
            copy_tile_to_dst_init_short(dfb::acc);
            copy_tile(dfb::acc, 0, DST_ACC);

            BINARY_OP(DST_IN, DST_ACC, DST_ACC);
            dfb_acc_obj.pop_front(ONE_TILE);

            dfb_in_obj.pop_front(ONE_TILE);

            tile_regs_commit();

            tile_regs_wait();

            dfb_out_obj.reserve_back(ONE_TILE);
            pack_reconfig_data_format(dfb::acc, dfb::dst);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_ACC, dfb::dst);
            dfb_out_obj.push_back(ONE_TILE);

            dfb_acc_obj.reserve_back(ONE_TILE);

            pack_reconfig_data_format(dfb::dst, dfb::acc);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_ACC, dfb::acc);

            tile_regs_release();

            dfb_acc_obj.push_back(ONE_TILE);
        }
    }

    // Clean-up and empty CB
    dfb_acc_obj.wait_front(ONE_TILE);
    dfb_acc_obj.pop_front(ONE_TILE);
}
