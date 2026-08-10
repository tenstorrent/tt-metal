// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.hpp"  // RandTile (owns rand_tile_init via init())

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t intermed_dfb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(intermed_dfb_id, intermed_dfb_id);

    eltwise_chain(
        IterationShape::tiles(num_tiles),
        RandTile<Dst::D0>{f2u_from.u, f2u_scale.u, seed},
        PackTile<output(intermed_dfb_id, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
}
