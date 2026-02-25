// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);

        tile_regs_acquire();
        copy_tile_init(cb_in);
        copy_tile(cb_in, 0, 0);

        sign_tile_init();
        sign_tile(0);

        tile_regs_commit();

        asm volatile("ebreak");

        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
}
