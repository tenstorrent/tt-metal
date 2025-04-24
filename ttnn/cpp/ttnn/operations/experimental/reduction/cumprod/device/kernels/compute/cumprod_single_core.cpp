// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_one = tt::CBIndex::c_16;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_24;

    constexpr uint32_t TILE_DEST = 0;
    constexpr uint32_t TILE_ACC = 1;

    constexpr uint32_t first_tile = 0;

    unary_op_init_common(cb_in, cb_out);

    cb_wait_front(cb_one, 1);

    for (unsigned i = 0; i < num_rows; i++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_one);
        copy_tile(cb_one, first_tile, TILE_DEST);
        tile_regs_commit();

        pack_reconfig_data_format(cb_intermed);
        tile_regs_wait();
        cb_reserve_back(cb_intermed, 1);
        pack_tile(TILE_DEST, cb_intermed);
        cb_push_back(cb_intermed, 1);
        tile_regs_release();

        for (unsigned j = 0; j < tiles_per_row; j++) {
            reconfig_data_format(cb_in, cb_intermed);
            cb_wait_front(cb_in, 1);
            // copy_tile_to_dst_init_short(cb_in);
            // copy_tile(cb_in, first_tile, TILE_DEST);

            cb_wait_front(cb_intermed, 1);
            // copy_tile_to_dst_init_short(cb_intermed);
            // copy_tile(cb_intermed, first_tile, TILE_ACC);

            tile_regs_acquire();

            mul_tiles_init(cb_in, cb_intermed);
            mul_tiles_bcast_rows(cb_in, cb_intermed, 0, 0, TILE_DEST);

            tile_regs_commit();

            cb_pop_front(cb_in, 1);
            cb_pop_front(cb_intermed, 1);

            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(TILE_DEST, cb_out);
            cb_push_back(cb_out, 1);

            cb_reserve_back(cb_intermed, 1);
            pack_reconfig_data_format(cb_intermed);
            pack_tile(TILE_DEST, cb_intermed);
            cb_push_back(cb_intermed, 1);

            tile_regs_release();
        }

        cb_wait_front(cb_intermed, 1);
        cb_pop_front(cb_intermed, 1);
    }

    cb_pop_front(cb_one, 1);
}

}  // namespace NAMESPACE
