// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    // output = input_a + scalar * input_b * input_c  (Int32 arithmetic)
    // D0=a, D1=b, D2=c, D3=scalar
    // IntMul(D3,D1,D3): D3 = scalar*b
    // IntMul(D3,D2,D2): D2 = scalar*b*c
    // IntAdd(D0,D2,D0): D0 = a + scalar*b*c
    unary_op_init_common(cb_in0, cb_out);

    auto chain = sfpu_chain(
        Load<cb_in0, Dst::D0>{},
        Load<cb_in1, Dst::D1>{},
        Load<cb_in2, Dst::D2>{},
        FillTileInt<Dst::D3>{scalar_arg},
        IntMul<DataFormat::Int32, Dst::D3, Dst::D1, Dst::D3>{},
        IntMul<DataFormat::Int32, Dst::D3, Dst::D2, Dst::D2>{},
        IntAdd<DataFormat::Int32, Dst::D0, Dst::D2, Dst::D0>{});

    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
}
