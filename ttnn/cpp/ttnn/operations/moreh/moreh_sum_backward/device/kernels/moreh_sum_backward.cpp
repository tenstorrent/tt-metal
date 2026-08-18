// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    DataflowBuffer dfb_in0_obj(dfb::in0);  // input
    DataflowBuffer dfb_in1_obj(dfb::in1);  // zero tile
    DataflowBuffer dfb_out0_obj(dfb::out0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    compute_kernel_hw_startup(dfb::in1, dfb::in0, dfb::out0);
    dfb_in1_obj.wait_front(onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        dfb_in0_obj.wait_front(onetile);
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init(dfb::in1, dfb::in0);
            add_tiles_bcast_scalar(dfb::in1, dfb::in0, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init(dfb::in1, dfb::in0);
            add_tiles_bcast_rows(dfb::in1, dfb::in0, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init(dfb::in1, dfb::in0);
            add_tiles_bcast_cols(dfb::in1, dfb::in0, 0, 0, dst0);
        } else {
            copy_tile_to_dst_init_short(dfb::in0);
            copy_tile(dfb::in0, 0, dst0);
        }
        tile_regs_commit();
        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, dfb::out0);
        tile_regs_release();
        dfb_out0_obj.push_back(onetile);
        dfb_in0_obj.pop_front(onetile);
    }
    dfb_in1_obj.pop_front(onetile);
}
