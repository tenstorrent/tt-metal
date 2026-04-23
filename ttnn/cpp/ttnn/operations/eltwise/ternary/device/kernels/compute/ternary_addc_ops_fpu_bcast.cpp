// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a (add operand)
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b (mul operand)
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c (mul operand)
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    // output = input_a + scalar * input_b * input_c (broadcast-aware)
    // BCAST_X: X is pre-waited externally per group → NoWaitNoPop / NoWaitNoPop.
    // Non-broadcast: WaitAndPop per tile.
#if BCAST_B
    constexpr auto policy_b = BinaryInputPolicy::NoWaitNoPop;
#else
    constexpr auto policy_b = BinaryInputPolicy::WaitAndPopPerTile;
#endif
#if BCAST_C
    constexpr auto policy_c = BinaryInputPolicy::NoWaitNoPop;
#else
    constexpr auto policy_c = BinaryInputPolicy::WaitAndPopPerTile;
#endif
#if BCAST_A
    constexpr auto policy_a = DestReuseInputPolicy::NoWaitNoPop;
#else
    constexpr auto policy_a = DestReuseInputPolicy::WaitAndPop;
#endif

    binary_op_init_common(cb_in1, cb_in2, cb_out);

    const bool scalar_is_not_1 = scalar_arg != 1u;

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

        if (scalar_is_not_1) {
            auto chain = sfpu_chain(
                FpuMul<cb_in1, cb_in2, Dst::D0, BroadcastDim::NONE, policy_b, policy_c>{},
                MulScalar<Dst::D0>{scalar_arg},
                DestReuseOp<
                    cb_in0,
                    EltwiseBinaryType::ELWADD,
                    EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                    Dst::D0,
                    policy_a>{});
            eltwise_op<cb_out>(chain, EltwiseTileShape::flat(freq - start));
        } else {
            auto chain = sfpu_chain(
                FpuMul<cb_in1, cb_in2, Dst::D0, BroadcastDim::NONE, policy_b, policy_c>{},
                DestReuseOp<
                    cb_in0,
                    EltwiseBinaryType::ELWADD,
                    EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                    Dst::D0,
                    policy_a>{});
            eltwise_op<cb_out>(chain, EltwiseTileShape::flat(freq - start));
        }

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

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_group(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        process_group(remaining_iterations, tile_start);
    }
}
