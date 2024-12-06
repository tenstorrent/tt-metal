// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // zero tile
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    cb_wait_front(cb_in1, onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_in0, onetile);
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init_short_with_dt(cb_in1, cb_in0);
            add_tiles_bcast_scalar(cb_in1, cb_in0, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init_short_with_dt(cb_in1, cb_in0);
            add_tiles_bcast_rows(cb_in1, cb_in0, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init_short_with_dt(cb_in1, cb_in0);
            add_tiles_bcast_cols(cb_in1, cb_in0, 0, 0, dst0);
        } else {
            copy_tile_init_with_dt(cb_in0);
            copy_tile(cb_in0, 0, dst0);
        }
        tile_regs_commit();

        cb_reserve_back(cb_intermed0, onetile);

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_intermed0);
        tile_regs_release();

        cb_push_back(cb_intermed0, onetile);
        cb_pop_front(cb_in0, onetile);

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
    cb_pop_front(cb_in1, onetile);
}
}  // namespace NAMESPACE
