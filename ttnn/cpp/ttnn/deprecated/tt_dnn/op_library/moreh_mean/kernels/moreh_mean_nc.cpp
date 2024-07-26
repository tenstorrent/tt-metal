// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_scalar = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1);

    cb_wait_front(cb_in1, onetile);
    cb_wait_front(cb_scalar, 1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);

            ACQ();
            cb_wait_front(cb_in0, onetile);
            if (enable_reload) {
                cb_wait_front(cb_intermed0, onetile);
            }

            uint32_t cb_add  = (enable_reload) ? (cb_intermed0) : (cb_in1);
            add_tiles_init();
            add_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);

            cb_pop_front(cb_in0, onetile);
            if (enable_reload) {
                cb_pop_front(cb_intermed0, onetile);
            }

            cb_reserve_back(cb_intermed0, onetile);
            pack_tile(dst0, cb_intermed0);
            cb_push_back(cb_intermed0, onetile);
            REL();

            enable_reload = true;
        }

        // output * (1 / number_of_elements)
        ACQ();
        cb_wait_front(cb_intermed0, onetile);
        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast<BroadcastType::SCALAR>(cb_intermed0, cb_scalar, 0, 0, 0);
        cb_reserve_back(cb_out0, onetile);
        pack_tile(dst0, cb_out0);
        cb_push_back(cb_out0, onetile);
        cb_pop_front(cb_intermed0, onetile);
        REL();
    }
}
}  // namespace NAMESPACE
