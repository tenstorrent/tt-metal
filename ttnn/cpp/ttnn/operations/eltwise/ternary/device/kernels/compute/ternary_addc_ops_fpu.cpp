// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a (add operand)
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b (mul operand)
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c (mul operand)
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    // output = input_a + scalar * input_b * input_c
    // FpuMul: D0 = b*c  (pops b and c immediately — FPU WaitAndPop default)
    // MulScalar: D0 *= scalar  (conditional, included always as scalar mul by 1 is harmless)
    // DestReuseOp<ELWADD,SRCA>: D0 = a + D0  (waits and pops a)
    binary_op_init_common(cb_in1, cb_in2, cb_out);

    const bool scalar_is_not_1 = scalar_arg != 1u;
    if (scalar_is_not_1) {
        auto chain = sfpu_chain(
            FpuMul<cb_in1, cb_in2, Dst::D0>{},
            MulScalar<Dst::D0>{scalar_arg},
            DestReuseOp<
                cb_in0,
                EltwiseBinaryType::ELWADD,
                EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                Dst::D0,
                DestReuseInputPolicy::WaitAndPop>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    } else {
        auto chain = sfpu_chain(
            FpuMul<cb_in1, cb_in2, Dst::D0>{},
            DestReuseOp<
                cb_in0,
                EltwiseBinaryType::ELWADD,
                EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                Dst::D0,
                DestReuseInputPolicy::WaitAndPop>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
}
