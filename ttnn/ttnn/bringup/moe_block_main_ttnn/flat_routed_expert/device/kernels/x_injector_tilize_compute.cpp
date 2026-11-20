// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat expert x injector tilizer (an x helper relay's TRISC). Each super-block (MT row tiles x SE_SBT K tiles of one
// virtual expert) arrives as MT row-major chunks of 32 rows x 32 tiles (x_injector_reader.cpp) and leaves as MT x 32
// bfp8 tiles, row tile major, for the relay's multicaster (x_injector_writer.cpp). CT: 0 RM_CB, 1 SB_CB, 2 MT; RT: 0
// super-blocks in total
#include <cstdint>
#ifndef SE_SBT
#define SE_SBT 32  // K tiles per super-block (the row-major chunk width / 32 columns)
#endif
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t rm_cb = get_compile_time_arg_val(0);
    constexpr uint32_t sb_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mt = get_compile_time_arg_val(2);
    cb_wait_front(tt::CBIndex::c_6, 1);  // this relay's super-block count (from x_injector_reader.cpp)
    const uint32_t num_sb = read_tile_value(tt::CBIndex::c_6, 0, 0);
    cb_pop_front(tt::CBIndex::c_6, 1);
    compute_kernel_hw_startup(rm_cb, sb_cb);
#ifdef SE_PACK_SRND  // stochastic rounding in the packer (the default rounds bf16 -> bfp8 ties away from zero)
    UNPACK((_llk_unpack_configure_stoch_rnd_<StochRndType::Pack>()));
#endif
    fast_tilize_init(rm_cb, SE_SBT, sb_cb);
    for (uint32_t b = 0; b < num_sb; ++b) {
        {
            cb_reserve_back(sb_cb, mt * SE_SBT);
        }
        for (uint32_t m = 0; m < mt; ++m) {
            {
                cb_wait_front(rm_cb, 32);
            }
            {
                fast_tilize_block(rm_cb, SE_SBT, sb_cb, 0, m * SE_SBT);
            }
            cb_pop_front(rm_cb, 32);
        }
        cb_push_back(sb_cb, mt * SE_SBT);
    }
    fast_tilize_uninit(rm_cb, sb_cb, SE_SBT);
}
