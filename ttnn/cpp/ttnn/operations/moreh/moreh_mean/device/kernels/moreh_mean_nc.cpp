// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    cb_wait_front(cb_in1, onetile);
    cb_wait_front(cb_scalar, 1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);

            tile_regs_acquire();
            cb_wait_front(cb_in0, onetile);
            if (enable_reload) {
                cb_wait_front(cb_intermed0, onetile);
            }

            uint32_t cb_add = (enable_reload) ? (cb_intermed0) : (cb_in1);
            add_tiles_init_with_dt(cb_in0, cb_add);
            add_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);

            cb_pop_front(cb_in0, onetile);
            if (enable_reload) {
                cb_pop_front(cb_intermed0, onetile);
            }
            tile_regs_commit();

            cb_reserve_back(cb_intermed0, onetile);
            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_intermed0);
            tile_regs_release();
            cb_push_back(cb_intermed0, onetile);

            enable_reload = true;
        }

        // output * (1 / number_of_elements)
        tile_regs_acquire();
        cb_wait_front(cb_intermed0, onetile);
        mul_tiles_bcast_scalar_init_short_with_dt(cb_intermed0, cb_scalar);
        mul_tiles_bcast<BroadcastType::SCALAR>(cb_intermed0, cb_scalar, 0, 0, 0);
        tile_regs_commit();

        cb_reserve_back(cb_out0, onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_out0);
        tile_regs_release();
        cb_push_back(cb_out0, onetile);
        cb_pop_front(cb_intermed0, onetile);
    }
}
}  // namespace NAMESPACE
