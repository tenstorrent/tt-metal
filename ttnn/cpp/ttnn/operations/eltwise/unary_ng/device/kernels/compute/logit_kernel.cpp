// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/compute_kernel_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;

    experimental::CircularBuffer cb_in(cb_input);
    experimental::CircularBuffer cb_out(cb_output);
    experimental::CircularBuffer cb_tmp(cb_tmp0);

    init_sfpu(cb_input, cb_output);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);
        cb_tmp.reserve_back(1);

        tile_regs_acquire();

        copy_tile_init(cb_input);
        copy_tile(cb_input, 0, 0);
#ifdef CLAMP
        clamp_tile_init();
        clamp_tile(0, packed_scalar1, packed_scalar2);
#endif
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_tmp0);
        tile_regs_release();

        cb_tmp.push_back(1);
        cb_tmp.wait_front(1);

        tile_regs_acquire();

        copy_tile_init(cb_tmp0);
        copy_tile(cb_tmp0, 0, 0);
        copy_tile(cb_tmp0, 0, 1);

        rsub_tile_init();
        rsub_tile(0, 0x3F800000u);  // 1.0 - x

        div_binary_tile_init();
        div_binary_tile(1, 0, 0);

        log_tile_init();
        log_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        cb_tmp.pop_front(1);
        cb_in.pop_front(1);
        cb_out.push_back(1);
    }
}
