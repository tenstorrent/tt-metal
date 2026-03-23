// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

#define APPROX false
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "experimental/circular_buffer.h"
#include "../accumulation_common.hpp"

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);
    const AccumulationOp accumulation_op = static_cast<AccumulationOp>(get_arg_val<uint32_t>(2));

    experimental::CircularBuffer cb_start_obj(cb_start);
    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_acc_obj(cb_acc);

    cb_start_obj.wait_front(ONE_TILE);

    for (uint32_t i = 0; i < num_rows; i++) {
        bool enable_reload = false;

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            tile_regs_acquire();
            const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
            cb_in_obj.wait_front(ONE_TILE);

            binary_op_init_common(cb_in, cb_op, cb_out);

#ifdef CUMSUM_USE_INT32
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, FIRST_TILE, INT32_TILE_DEST);

            copy_tile_to_dst_init_short(cb_op);
            copy_tile(cb_op, FIRST_TILE, INT32_TILE_ACC);
#endif  // CUMSUM_USE_INT32

            // cumulating tiles along the first dimension,
            // data is not dependent on itself within tiles
            if (accumulation_op == AccumulationOp::CUMPROD) {
                mul_tiles_init(cb_in, cb_op);
                mul_tiles(cb_in, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);
            } else if (accumulation_op == AccumulationOp::CUMSUM) {
#ifdef CUMSUM_USE_INT32
                add_int_tile_init();
                add_int_tile<DataFormat::Int32>(INT32_TILE_DEST, INT32_TILE_ACC, INT32_TILE_DEST);
#else
                add_tiles_init(cb_in, cb_op);
                add_tiles(cb_in, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);
#endif  // CUMSUM_USE_INT32
            }

            cb_in_obj.pop_front(ONE_TILE);
            if (enable_reload) {
                cb_acc_obj.pop_front(ONE_TILE);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_acc_obj.reserve_back(ONE_TILE);
            pack_tile(WORKING_REG, cb_acc);
            cb_acc_obj.push_back(ONE_TILE);

            cb_acc_obj.wait_front(ONE_TILE);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, FIRST_TILE, WORKING_REG);

            cb_out_obj.reserve_back(ONE_TILE);
            pack_tile(WORKING_REG, cb_out);
            cb_out_obj.push_back(ONE_TILE);

            tile_regs_release();

            enable_reload = true;
        }

        cb_acc_obj.pop_front(ONE_TILE);
    }

    cb_start_obj.pop_front(ONE_TILE);
}
