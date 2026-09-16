// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.hpp"  // RandTile (owns rand_tile_init via init())

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(0);

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

    compute_kernel_hw_startup(output_dfb_id, output_dfb_id);

    eltwise_chain(
        IterationShape::tiles(num_tiles),
        RandTile<Dst::D0>{f2u_lower_bound.u, f2u_scale.u, seed, start_id},
        PackTile<output(output_dfb_id, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
}
