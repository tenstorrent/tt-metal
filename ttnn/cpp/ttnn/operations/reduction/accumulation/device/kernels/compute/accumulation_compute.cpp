// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#define APPROX false
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "experimental/circular_buffer.h"
#include "../accumulation_common.hpp"

#include "api/debug/dprint_tensix.h"

void kernel_main() {
    using namespace ckernel;  // DEBUG

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    experimental::CircularBuffer cb_start_obj(cb_start);
    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_acc_obj(cb_acc);

    binary_op_init_common(cb_in, cb_acc, cb_out);

#ifdef USE_FPU
    BINARY_OP_INIT(cb_in, cb_acc);
#else
    BINARY_OP_INIT();
#endif  // USE_FPU

    DPRINT << "default acc value = " << DEFAULT_ACC_VALUE << ENDL();
    DPRINT << "tiles per row = " << tiles_per_row << ENDL();
    DPRINT << "num rows = " << num_rows << ENDL();
    DPRINT << "cb in = " << cb_in << ENDL();

    constexpr uint32_t DST_IN = 0;
    constexpr uint32_t DST_ACC = 1;

    for (uint32_t i = 0; i < num_rows; i++) {
        DPRINT << "i = " << i << ENDL();

        cb_acc_obj.wait_front(ONE_TILE);

        // Fill cb_acc to default values
        tile_regs_acquire();

        fill_tile_init();
        fill_tile(DST_ACC, DEFAULT_ACC_VALUE);  // TODO: Check with int32 multiply
        tile_regs_commit();
        cb_acc_obj.pop_front(ONE_TILE);

        cb_acc_obj.reserve_back(ONE_TILE);

        // Pack to acc
        tile_regs_wait();
        pack_reconfig_data_format(cb_acc);
        pack_tile<true>(DST_ACC, cb_acc, 0);
        tile_regs_release();

        cb_acc_obj.push_back(ONE_TILE);

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            UNPACK(DPRINT << "j = " << j << ENDL(););

            // Synchronize unpacker-packer

            cb_acc_obj.reserve_back(ONE_TILE);
            cb_acc_obj.push_back(ONE_TILE);

            cb_acc_obj.wait_front(ONE_TILE);

            tile_regs_acquire();
            cb_in_obj.wait_front(ONE_TILE);

#ifdef USE_FPU
            reconfig_data_format(cb_in, cb_acc);
            BINARY_OP(cb_in, cb_acc, 0, 0, DST_ACC);
#else
            reconfig_data_format(cb_in, cb_in);
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, 0, DST_IN);

            // UNPACK(
            //	DPRINT << "cb in = " << ENDL();
            //	);
            // dprint_tensix_dest_reg(DST_IN);

            // TODO: reconfig buffer for fp32 acc if using bfloat16 or float32

            reconfig_data_format(cb_acc, cb_acc);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 0, DST_ACC);

            // UNPACK(
            //	DPRINT << "cb acc = " << ENDL();
            //	);
            // dprint_tensix_dest_reg(DST_ACC);

            BINARY_OP(DST_IN, DST_ACC, DST_ACC);
#endif  // USE_FPU

            cb_in_obj.pop_front(ONE_TILE);

            tile_regs_commit();

            tile_regs_wait();

            cb_out_obj.reserve_back(ONE_TILE);
            pack_reconfig_data_format(cb_acc, cb_out);  // If using fp32_acc_to_dest
            pack_tile(DST_ACC, cb_out);
            cb_out_obj.push_back(ONE_TILE);

            pack_reconfig_data_format(cb_out, cb_acc);  // If using fp32_acc_to_dest

            // out_of_order_output to accumulate to not increment cb_acc write 'cursor'
            pack_tile<true>(DST_ACC, cb_acc, 0);

            tile_regs_release();

            cb_acc_obj.pop_front(ONE_TILE);
        }
    }
    DPRINT << "done" << ENDL();
}
