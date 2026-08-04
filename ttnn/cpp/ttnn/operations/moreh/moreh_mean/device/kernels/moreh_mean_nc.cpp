// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);

    DataflowBuffer dfb_in0_obj(dfb::input);
    DataflowBuffer dfb_in1_obj(dfb::in1);
    DataflowBuffer dfb_scalar_obj(dfb::scalar);
    DataflowBuffer dfb_out0_obj(dfb::out);
    DataflowBuffer dfb_intermed0_obj(dfb::intermed0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(dfb::input, dfb::in1, dfb::out);

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

            uint32_t cb_add = (enable_reload) ? (dfb::intermed0) : (dfb::in1);
            add_tiles_init_with_dt(dfb_in0_obj, DataflowBuffer(cb_add));
            add_tiles(dfb::input, cb_add, first_tile, first_tile, dst0);

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
        mul_tiles_bcast<BroadcastType::SCALAR>(dfb::intermed0, dfb::scalar, 0, 0, 0);
        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_out0_obj);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
        dfb_intermed0_obj.pop_front(onetile);
    }
}
