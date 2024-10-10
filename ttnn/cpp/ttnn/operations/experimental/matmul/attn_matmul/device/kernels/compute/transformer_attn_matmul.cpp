// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t onetile = 1;

    constexpr uint32_t transpose_hw = get_compile_time_arg_val(0);
    uint32_t batch = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_intermed0 = 24;
    constexpr uint32_t cb_intermed1 = 25;
    constexpr uint32_t cb_intermed2 = 26;
    constexpr uint32_t out_cb_id = 16;

    constexpr uint32_t num_rows_in_one_tile = 32;

    mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);

    for (uint32_t nb = 0; nb < batch; ++nb)
    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) // output tile index of C
    {
        for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; ++tile_row_id) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                if (tile_row_id == 0) {
                    cb_wait_front(cb_in0, kt+1);
                }
                cb_wait_front(cb_in1, onetile);

                matmul_tiles(cb_in0, cb_in1, kt, 0, 0, transpose_hw);

                cb_pop_front(cb_in1, onetile);
            }
            tile_regs_commit();

            cb_reserve_back(cb_intermed0, onetile);
            tile_regs_wait();
            pack_tile(0, cb_intermed0);
            tile_regs_release();
            cb_push_back(cb_intermed0, onetile);

            // untilize tile and write to CB::c_intermed1
            reconfig_data_format_srca(cb_in1, cb_intermed0);
            cb_wait_front(cb_intermed0, onetile);
            untilize_init_short(cb_intermed0);
            cb_reserve_back(cb_intermed1, onetile);
            untilize_block(cb_intermed0, onetile, cb_intermed1);
            cb_push_back(cb_intermed1, onetile);

            cb_pop_front(cb_intermed0, onetile);
            untilize_uninit(cb_intermed0);

            reconfig_data_format_srca(cb_intermed0, cb_in1);
            mm_init_short(cb_in0, cb_in1, transpose_hw);
        }
        cb_pop_front(cb_in0, Kt);

        // cb_intermed2 comes from reader; untilized row-major tile
        pack_reconfig_data_format(cb_intermed1, out_cb_id);
        cb_wait_front(cb_intermed2, onetile);
        cb_reserve_back(out_cb_id, onetile);

        // tilize CB::intermed2 and write to CB::c_out0
        tilize_init_short_with_dt(cb_in1, cb_intermed2, onetile);
        tilize_block(cb_intermed2, onetile, out_cb_id);
        cb_push_back(out_cb_id, onetile);

        cb_pop_front(cb_intermed2, onetile);
        tilize_uninit(cb_intermed2);

        pack_reconfig_data_format(out_cb_id, cb_intermed0);
        mm_init_short_with_dt(cb_in0, cb_in1, cb_intermed2, transpose_hw);
    }

}
} // NAMESPACE
