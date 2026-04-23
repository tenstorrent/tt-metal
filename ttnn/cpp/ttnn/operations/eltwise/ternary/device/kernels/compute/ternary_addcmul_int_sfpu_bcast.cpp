// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tile_freq = get_arg_val<uint32_t>(1);
    const uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    // output = input_a + scalar * input_b * input_c  (Int32, broadcast-aware)
    // Broadcast inputs use NoWaitNoPop (waited/popped externally per frequency group).
    // Non-broadcast inputs use WaitAndPop (per-tile streaming).
#if BCAST_A
    constexpr auto policy_a = LoadPolicy::NoWaitNoPop;
#else
    constexpr auto policy_a = LoadPolicy::WaitAndPop;
#endif
#if BCAST_B
    constexpr auto policy_b = LoadPolicy::NoWaitNoPop;
#else
    constexpr auto policy_b = LoadPolicy::WaitAndPop;
#endif
#if BCAST_C
    constexpr auto policy_c = LoadPolicy::NoWaitNoPop;
#else
    constexpr auto policy_c = LoadPolicy::WaitAndPop;
#endif

    unary_op_init_common(cb_in0, cb_out);

    auto chain = sfpu_chain(
        Load<cb_in0, Dst::D0, policy_a>{},
        Load<cb_in1, Dst::D1, policy_b>{},
        Load<cb_in2, Dst::D2, policy_c>{},
        FillTileInt<Dst::D3>{scalar_arg},
        IntMul<DataFormat::Int32, Dst::D3, Dst::D1, Dst::D3>{},
        IntMul<DataFormat::Int32, Dst::D3, Dst::D2, Dst::D2>{},
        IntAdd<DataFormat::Int32, Dst::D0, Dst::D2, Dst::D0>{});

    // Replicate the tile_freq / tile_start loop from the original kernel.
    // Broadcast CBs are waited/popped externally around each frequency group.
    auto process_group = [&](uint32_t freq, uint32_t start) {
#if BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(freq - start));
#if BCAST_A
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
    };

    uint32_t ts = tile_start;
    uint32_t complete_iterations = (num_tiles + ts) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + ts) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, ts = 0) {
        process_group(tile_freq, ts);
    }
    if (remaining_iterations > 0) {
        process_group(remaining_iterations, ts);
    }
}
