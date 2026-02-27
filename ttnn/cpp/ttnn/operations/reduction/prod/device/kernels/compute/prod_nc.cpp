// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr auto cb_intermed0 = tt::CBIndex::c_2;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_intermed0_obj(cb_intermed0);
    experimental::CircularBuffer cb_out0_obj(cb_out0);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_in1_obj.wait_front(onetile);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);
            uint32_t cb_add = (enable_reload) ? (cb_intermed0) : (cb_in1);

            cb_in0_obj.wait_front(onetile);
            if (enable_reload) {
                cb_intermed0_obj.wait_front(onetile);
            }

            tile_regs_acquire();
            mul_tiles_init(cb_in0, cb_add);
            mul_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);
            tile_regs_commit();

            cb_in0_obj.pop_front(onetile);
            if (enable_reload) {
                cb_intermed0_obj.pop_front(onetile);
            }

            auto& cb_out_obj = last_out ? cb_out0_obj : cb_intermed0_obj;
            cb_out_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_tile(dst0, cb_out_obj.get_cb_id());
            tile_regs_release();
            cb_out_obj.push_back(onetile);
            enable_reload = true;
        }
    }
}
