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
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"

#define APPROX false
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "experimental/circular_buffer.h"
#include "../accumulation_common.hpp"

void kernel_main() {
    const float default_acc_value = norm::kernel_util::generic::bit_cast<float>(get_compile_time_arg_val(0));

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_acc_obj(cb_acc);  // note: only used in compute kernel

    BINARY_OP_INIT();

    constexpr uint32_t DST_IN = 0;
    constexpr uint32_t DST_ACC = 1;

    cb_acc_obj.reserve_back(ONE_TILE);
    cb_acc_obj.push_back(ONE_TILE);

    for (uint32_t i = 0; i < num_rows; i++) {
        // Synchronize unpacker-packer between iterations
        // This is necessary to avoid data-races on cb_acc
        cb_acc_obj.wait_front(ONE_TILE);
        cb_acc_obj.pop_front(ONE_TILE);
        unary_op_init_common(cb_acc, cb_acc);

        tile_regs_acquire();
        reconfig_data_format(cb_acc, cb_acc);

        fill_tile_init();
        fill_tile(DST_ACC, default_acc_value);

        tile_regs_commit();

        tile_regs_wait();

        // out_of_order_output to keep packing to cb_acc at the same location
        pack_reconfig_data_format(cb_acc);
        pack_tile(DST_ACC, cb_acc);
        tile_regs_release();

        binary_op_init_common(cb_in, cb_acc, cb_out);

        cb_acc_obj.reserve_back(ONE_TILE);
        cb_acc_obj.push_back(ONE_TILE);

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            // Synchronize unpacker-packer between iterations
            cb_acc_obj.wait_front(ONE_TILE);
            cb_acc_obj.pop_front(ONE_TILE);

            tile_regs_acquire();
            cb_in_obj.wait_front(ONE_TILE);

            reconfig_data_format(cb_in, cb_in);
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, 0, DST_IN);

            reconfig_data_format(cb_acc, cb_acc);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 0, DST_ACC);

            BINARY_OP(DST_IN, DST_ACC, DST_ACC);

            cb_in_obj.pop_front(ONE_TILE);

            tile_regs_commit();

            tile_regs_wait();

            cb_out_obj.reserve_back(ONE_TILE);
            pack_reconfig_data_format(cb_acc, cb_out);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_ACC, cb_out);
            cb_out_obj.push_back(ONE_TILE);

            pack_reconfig_data_format(cb_out, cb_acc);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_ACC, cb_acc);

            tile_regs_release();

            cb_acc_obj.reserve_back(ONE_TILE);
            cb_acc_obj.push_back(ONE_TILE);
        }
    }

    // Clean-up and empty CB
    cb_acc_obj.wait_front(ONE_TILE);
    cb_acc_obj.pop_front(ONE_TILE);
}
