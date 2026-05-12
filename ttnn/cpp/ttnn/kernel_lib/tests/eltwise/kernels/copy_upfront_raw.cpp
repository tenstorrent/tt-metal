// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Raw LLK reference for the upfront-block lifecycle. Hand-coded equivalent of the
// chain pipeline's `WaitUpfrontPopAtEnd + UpfrontReservePushAtEnd` path.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifndef UPFRONT_N
#define UPFRONT_N 4
#endif

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t upfront_n = UPFRONT_N;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);
    exp_tile_init<>();

    cb_wait_front(cb_in, upfront_n);
    cb_reserve_back(cb_out, upfront_n);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(cb_in, i, 0);
        exp_tile<>(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out, i);
        tile_regs_release();
    }

    cb_pop_front(cb_in, upfront_n);
    cb_push_back(cb_out, upfront_n);
}
