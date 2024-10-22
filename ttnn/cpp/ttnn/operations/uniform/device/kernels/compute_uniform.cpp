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
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t end_id = start_id + num_tiles;

    init_sfpu(intermed_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        rand_tile_init(i * seed);

        cb_reserve_back(intermed_cb_id, 1);

        tile_regs_acquire();
        rand_tile(0, f2u_from.f, f2u_to.f);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, intermed_cb_id, 0);
        tile_regs_release();

        cb_push_back(intermed_cb_id, 1);
    }
}
}  // namespace NAMESPACE
