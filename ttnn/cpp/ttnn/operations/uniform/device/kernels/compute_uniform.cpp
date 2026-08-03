// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale, f2u_upper_bound;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    // Construct nextafter(to, -infinity) directly, then choose the largest
    // scale whose rounded upper endpoint remains below `to`. This avoids a
    // per-row clamp in the SFPU hot loop. Validation guarantees from < to.
    uint32_t upper_bound_bits;
    if ((f2u_to.u & 0x7FFFFFFFU) == 0) {
        upper_bound_bits = 0x80000001U;
    } else {
        upper_bound_bits = (f2u_to.u >> 31) ? f2u_to.u + 1U : f2u_to.u - 1U;
    }
    f2u_upper_bound.u = upper_bound_bits;
    f2u_scale.f = f2u_upper_bound.f - f2u_from.f;
    if (!(f2u_from.f + f2u_scale.f < f2u_to.f) && f2u_scale.u != 0) {
        --f2u_scale.u;
    }
    const uint32_t start_id = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t end_id = start_id + num_tiles;

    CircularBuffer cb_intermed(intermed_cb_id);

    init_sfpu(intermed_cb_id, intermed_cb_id);

    // The host gives neighbouring cores nearby seeds. start_id is unique for
    // every participating core; an odd Weyl multiplier spreads those related
    // LFSR starting states over 32 bits.
    constexpr uint32_t core_seed_multiplier = 0x9E3779B9U;
    rand_tile_init(seed + start_id * core_seed_multiplier);
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, f2u_from.u, f2u_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, intermed_cb_id, 0);
        tile_regs_release();

        cb_intermed.push_back(1);
    }
}
