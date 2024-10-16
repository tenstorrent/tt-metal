// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() {
    acquire_dst();
}
ALWI void REL() {
    release_dst();
}

namespace NAMESPACE {
void MAIN {
    const auto num_tiles_to_cumsum = get_arg_val<uint32_t>(0);
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(cb_in0, cb_in1);
    cb_wait_front(cb_in1, onetile);

    for (uint32_t i = 0; i < num_output_tiles_per_core; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_tiles_to_cumsum; ++j) {
            ACQ();
            uint32_t cb_add = (enable_reload) ? (cb_intermed0) : (cb_in1);
            cb_wait_front(cb_in0, onetile);

            add_tiles_init();
            add_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);

            cb_pop_front(cb_in0, onetile);
            if (enable_reload) {
                cb_pop_front(cb_intermed0, onetile);
            }

            // pack to intermed0
            cb_reserve_back(cb_intermed0, onetile);
            pack_tile(dst0, cb_intermed0);
            cb_push_back(cb_intermed0, onetile);
            REL();

            // copy tile from intermed0 to out0
            ACQ();
            cb_wait_front(cb_intermed0, onetile);
            copy_tile_to_dst_init_short();
            copy_tile(tt::CB::c_intermed0, first_tile, dst0);
            cb_reserve_back(cb_out0, onetile);
            pack_tile(dst0, cb_out0);
            cb_push_back(cb_out0, onetile);
            REL();
            enable_reload = true;
        }
        cb_pop_front(cb_intermed0, onetile);
    }
    cb_pop_front(cb_in1, onetile);
}
}  // namespace NAMESPACE
