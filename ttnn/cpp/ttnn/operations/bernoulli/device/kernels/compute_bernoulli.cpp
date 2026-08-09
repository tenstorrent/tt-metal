// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t intermed_dfb_id = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(intermed_dfb_id, intermed_dfb_id);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    union f2u {
        float f;
        uint32_t u;
    } rand_scale;
    rand_scale.f = 1;
    uint32_t rand_from = 0;

    eltwise_chain(
        EltwiseShape::tiles(num_tiles),
        RandTile<Dst::D0>{rand_from, rand_scale.u, seed},
        PackTile<output(intermed_dfb_id, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{});
}
