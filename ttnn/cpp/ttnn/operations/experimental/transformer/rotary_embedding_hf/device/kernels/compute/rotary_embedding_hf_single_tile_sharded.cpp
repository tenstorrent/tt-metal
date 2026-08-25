// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t heads_per_batch_t = get_arg(args::heads_per_batch_t);
    constexpr uint32_t batch_per_core = get_arg(args::batch_per_core);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_trans_mat(dfb::trans_mat);
    DataflowBuffer dfb_rotated_in_interm(dfb::rotated_in_interm);
    DataflowBuffer dfb_cos_interm(dfb::cos_interm);
    DataflowBuffer dfb_sin_interm(dfb::sin_interm);
    DataflowBuffer dfb_out(dfb::out);

    dfb_trans_mat.wait_front(onetile);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in, dfb::trans_mat, dfb::rotated_in_interm);
    matmul_init(dfb::in, dfb::trans_mat);
    compute_kernel_hw_startup(dfb::rotated_in_interm, dfb::sin, dfb::sin_interm);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        dfb_sin.reserve_back(onetile);
        dfb_cos.reserve_back(onetile);
        dfb_sin.push_back(onetile);
        dfb_cos.push_back(onetile);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            dfb_rotated_in_interm.reserve_back(onetile);
            dfb_sin_interm.reserve_back(onetile);
            dfb_cos_interm.reserve_back(onetile);
            dfb_out.reserve_back(onetile);

            dfb_in.reserve_back(onetile);
            dfb_in.push_back(onetile);
            dfb_in.wait_front(onetile);

            reconfig_data_format(dfb::in, dfb::trans_mat);
            pack_reconfig_data_format(dfb::rotated_in_interm);
            matmul_init(dfb::in, dfb::trans_mat);
            tile_regs_acquire();
            matmul_tiles(dfb::in, dfb::trans_mat, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::rotated_in_interm);
            tile_regs_release();
            dfb_rotated_in_interm.push_back(onetile);

            dfb_rotated_in_interm.wait_front(onetile);
            dfb_sin.wait_front(onetile);
            reconfig_data_format(dfb::rotated_in_interm, dfb::sin);
            pack_reconfig_data_format(dfb::sin_interm);
            tile_regs_acquire();
            mul_bcast_rows_init(dfb::rotated_in_interm, dfb::sin);
            mul_tiles_bcast_rows(dfb::rotated_in_interm, dfb::sin, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::sin_interm);
            tile_regs_release();
            dfb_sin_interm.push_back(onetile);
            dfb_rotated_in_interm.pop_front(onetile);

            dfb_cos.wait_front(onetile);
            reconfig_data_format(dfb::in, dfb::cos);
            pack_reconfig_data_format(dfb::cos_interm);
            tile_regs_acquire();
            mul_bcast_rows_init(dfb::in, dfb::cos);
            mul_tiles_bcast_rows(dfb::in, dfb::cos, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::cos_interm);
            tile_regs_release();
            dfb_cos_interm.push_back(onetile);
            dfb_in.pop_front(onetile);

            dfb_cos_interm.wait_front(onetile);
            dfb_sin_interm.wait_front(onetile);
            reconfig_data_format(dfb::cos_interm, dfb::sin_interm);
            pack_reconfig_data_format(dfb::out);
            add_init(dfb::cos_interm, dfb::sin_interm);
            tile_regs_acquire();
            add_tiles(dfb::cos_interm, dfb::sin_interm, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();
            dfb_out.push_back(onetile);
            dfb_cos_interm.pop_front(onetile);
            dfb_sin_interm.pop_front(onetile);
        }

        dfb_sin.pop_front(onetile);
        dfb_cos.pop_front(onetile);
    }
}
