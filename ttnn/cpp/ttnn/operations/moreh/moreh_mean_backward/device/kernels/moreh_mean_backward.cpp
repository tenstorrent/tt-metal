// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/kernel_args.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    // compile-time args
    constexpr auto num_output_tiles = get_arg(args::num_output_tiles);
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    DataflowBuffer dfb_in0_obj(dfb::in);         // input
    DataflowBuffer dfb_in1_obj(dfb::zero);       // zero tile
    DataflowBuffer dfb_scalar_obj(dfb::scalar);  // 1/num_dim bcast scalar
    DataflowBuffer dfb_out0_obj(dfb::out);       // output
    DataflowBuffer dfb_intermed0_obj(dfb::intermed);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    compute_kernel_hw_startup(dfb::in, dfb::zero, dfb::out);
    dfb_in1_obj.wait_front(onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        dfb_in0_obj.wait_front(onetile);
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init_with_dt(dfb_in1_obj, dfb_in0_obj);
            add_tiles_bcast_scalar(dfb::zero, dfb::in, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init_with_dt(dfb_in1_obj, dfb_in0_obj);
            add_tiles_bcast_rows(dfb::zero, dfb::in, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init_with_dt(dfb_in1_obj, dfb_in0_obj);
            add_tiles_bcast_cols(dfb::zero, dfb::in, 0, 0, dst0);
        } else {
            copy_tile_init_with_dt(dfb_in0_obj);
            copy_tile(dfb::in, 0, dst0);
        }
        tile_regs_commit();

        dfb_intermed0_obj.reserve_back(onetile);

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_intermed0_obj);
        tile_regs_release();

        dfb_intermed0_obj.push_back(onetile);
        dfb_in0_obj.pop_front(onetile);

        // output * (1 / number_of_elements)
        tile_regs_acquire();
        dfb_intermed0_obj.wait_front(onetile);
        mul_bcast_scalar_init_with_dt(dfb_intermed0_obj, dfb_scalar_obj);
        mul_tiles_bcast<BroadcastType::SCALAR>(dfb::intermed, dfb::scalar, 0, 0, 0);
        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_out0_obj);
        tile_regs_release();

        dfb_out0_obj.push_back(onetile);
        dfb_intermed0_obj.pop_front(onetile);
    }
    dfb_in1_obj.pop_front(onetile);
}
