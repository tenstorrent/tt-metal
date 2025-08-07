// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define APPROX false
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "../accumulation_common.hpp"

namespace NAMESPACE {
void MAIN {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(1);
    const AccumulationOp accumulation_op = static_cast<AccumulationOp>(get_arg_val<uint32_t>(2));

    cb_wait_front(cb_start, ONE_TILE);

    for (uint32_t i = 0; i < num_rows; i++) {
        bool enable_reload = false;

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            tile_regs_acquire();
            const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
            cb_wait_front(cb_in, ONE_TILE);

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
                add_int32_tile(INT32_TILE_DEST, INT32_TILE_ACC);
#else
                add_tiles_init(cb_in, cb_op);
                add_tiles(cb_in, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);
#endif  // CUMSUM_USE_INT32
            }

            cb_pop_front(cb_in, ONE_TILE);
            if (enable_reload) {
                cb_pop_front(cb_acc, ONE_TILE);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_acc, ONE_TILE);
            // pack_reconfig_data_format(cb_acc);
            pack_tile(WORKING_REG, cb_acc);
            cb_push_back(cb_acc, ONE_TILE);

            // release_dst();

            // acquire_dst();

            cb_wait_front(cb_acc, ONE_TILE);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, FIRST_TILE, WORKING_REG);

            cb_reserve_back(cb_out, ONE_TILE);
            // pack_reconfig_data_format(cb_out);
            pack_tile(WORKING_REG, cb_out);
            cb_push_back(cb_out, ONE_TILE);

            tile_regs_release();

            enable_reload = true;
        }

        // cb_wait_front(cb_acc);
        cb_pop_front(cb_acc, ONE_TILE);
    }

    cb_pop_front(cb_start, ONE_TILE);
}

}  // namespace NAMESPACE
