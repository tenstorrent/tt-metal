// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rand_uint.h"

namespace NAMESPACE {

void MAIN {
    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t end_id = start_id + num_tiles;

    constexpr auto cb_intermed0_id = tt::CB::c_intermed0;

    unary_op_init_common(cb_intermed0_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_intermed0_id, 1);
        rand_uint_tile_init(i * seed);

        tile_regs_acquire();
        rand_uint_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_intermed0_id);
        tile_regs_release();

        cb_push_back(cb_intermed0_id, 1);
    }
}
}  // namespace NAMESPACE
