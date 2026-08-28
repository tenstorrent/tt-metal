// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_lower_bound, f2u_upper_bound, f2u_scale;
    f2u_lower_bound.u = get_arg_val<uint32_t>(1);
    f2u_upper_bound.u = get_arg_val<uint32_t>(2);
    // The host supplies inclusive endpoints that are representable in the
    // destination dtype. Choose the largest scale whose rounded endpoint does
    // not exceed the upper bound, avoiding a clamp in the SFPU hot loop.
    f2u_scale.f = f2u_upper_bound.f - f2u_lower_bound.f;
    if (f2u_lower_bound.f + f2u_scale.f > f2u_upper_bound.f && f2u_scale.u != 0) {
        --f2u_scale.u;
    }
    const uint32_t start_id = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t end_id = start_id + num_tiles;

    CircularBuffer cb_output(output_cb_id);

    compute_kernel_hw_startup(output_cb_id, output_cb_id);
    copy_init(output_cb_id);

    rand_tile_init(seed, start_id);
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_output.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, f2u_lower_bound.u, f2u_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, output_cb_id, 0);
        tile_regs_release();

        cb_output.push_back(1);
    }
}
