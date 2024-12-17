// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rand.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t end_id = start_id + num_tiles;

    init_sfpu(intermed_cb_id, intermed_cb_id);

    union f2u {
        float f;
        uint32_t u;
    } rand_scale;
    rand_scale.f = 1;
    uint32_t rand_from = 0;

    rand_tile_init(seed);
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(intermed_cb_id, 1);

        tile_regs_acquire();
        rand_tile(0, rand_from, rand_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, intermed_cb_id, 0);
        tile_regs_release();

        cb_push_back(intermed_cb_id, 1);
    }
}
}  // namespace NAMESPACE
