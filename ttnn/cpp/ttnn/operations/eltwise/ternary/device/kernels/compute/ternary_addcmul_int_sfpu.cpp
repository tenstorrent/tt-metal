// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    static_assert(num_tiles_per_cycle == 1, "addcmul_int path runs one tile per chain invocation");

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(cb_in0, cb_out);
    ckernel::init_sfpu(cb_in0, cb_out);

    using compute_kernel_lib::AddIntBinary;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;
    using compute_kernel_lib::FillInt;
    using compute_kernel_lib::MulIntBinary;

    // Chain mirrors the original sequence:
    //   D0=a, D1=b, D2=c, D3=fill(scalar)
    //   mul_int(D3, D1, D3) => D3 = scalar*b
    //   mul_int(D3, D2, D2) => D2 = scalar*b*c
    //   add_int(D0, D2, D0) => D0 = a + scalar*b*c
    eltwise_pipeline<cb_out>(
        num_tiles,
        eltwise_chain(
            CopyTile<cb_in0, Dst::D0>{},
            CopyTile<cb_in1, Dst::D1>{},
            CopyTile<cb_in2, Dst::D2>{},
            FillInt<ADDCMUL_DATA_FORMAT, Dst::D3>{{}, scalar_arg},
            MulIntBinary<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D1, Dst::D3>{},
            MulIntBinary<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D2, Dst::D2>{},
            AddIntBinary<ADDCMUL_DATA_FORMAT, Dst::D0, Dst::D2, Dst::D0>{}));
}
