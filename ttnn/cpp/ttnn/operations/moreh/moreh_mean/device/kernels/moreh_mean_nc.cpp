// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    DataflowBuffer dfb_in1_obj(cb_in1);
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    DataflowBuffer dfb_scalar_obj(cb_scalar);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    DataflowBuffer dfb_intermed0_obj(cb_intermed0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    dfb_in1_obj.wait_front(onetile);
    dfb_scalar_obj.wait_front(1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);

            tile_regs_acquire();
            dfb_in0_obj.wait_front(onetile);
            if (enable_reload) {
                dfb_intermed0_obj.wait_front(onetile);
            }

            uint32_t cb_add = (enable_reload) ? (cb_intermed0) : (cb_in1);
            add_tiles_init_with_dt(dfb_in0_obj, DataflowBuffer(cb_add));
            add_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);

            dfb_in0_obj.pop_front(onetile);
            if (enable_reload) {
                dfb_intermed0_obj.pop_front(onetile);
            }
            tile_regs_commit();

            dfb_intermed0_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_intermed0_obj);
            tile_regs_release();
            dfb_intermed0_obj.push_back(onetile);

            enable_reload = true;
        }

        // output * (1 / number_of_elements)
        tile_regs_acquire();
        dfb_intermed0_obj.wait_front(onetile);
        mul_tiles_bcast_scalar_init_short_with_dt(dfb_intermed0_obj, dfb_scalar_obj);
        mul_tiles_bcast<BroadcastType::SCALAR>(cb_intermed0, cb_scalar, 0, 0, 0);
        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_out0_obj);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
        dfb_intermed0_obj.pop_front(onetile);
    }
}
