// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include "compute_kernel_api/common.h"

#include "../cumprod_common.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_in, cb_one, cb_out);

    cb_wait_front(cb_one, ONE_TILE);

    for (uint32_t i = 0; i < num_rows; i++) {
        bool enable_reload = false;

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            acquire_dst();
            uint32_t cb_mul = enable_reload ? cb_intermed : cb_one;
            cb_wait_front(cb_in, ONE_TILE);

            // multiplying tiles along the first dimension,
            // data is not dependent on itself within tiles
            mul_tiles_init(cb_in, cb_mul);
            mul_tiles(cb_in, cb_mul, FIRST_TILE, FIRST_TILE, WORKING_REG);

            cb_pop_front(cb_in, ONE_TILE);
            if (enable_reload) {
                cb_pop_front(cb_intermed, ONE_TILE);
            }

            cb_reserve_back(cb_intermed, ONE_TILE);
            pack_reconfig_data_format(cb_intermed);
            pack_tile(WORKING_REG, cb_intermed);
            cb_push_back(cb_intermed, ONE_TILE);

            release_dst();

            acquire_dst();

            cb_wait_front(cb_intermed, ONE_TILE);
            copy_tile_to_dst_init_short(cb_intermed);
            copy_tile(cb_intermed, FIRST_TILE, WORKING_REG);

            cb_reserve_back(cb_out, ONE_TILE);
            pack_reconfig_data_format(cb_out);
            pack_tile(WORKING_REG, cb_out);
            cb_push_back(cb_out, ONE_TILE);

            release_dst();

            enable_reload = true;
        }

        cb_pop_front(cb_intermed, ONE_TILE);
    }

    cb_pop_front(cb_one, ONE_TILE);
}

}  // namespace NAMESPACE
