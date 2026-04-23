// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    // tanhshrink(x) = x - tanh(x)
    // INP_FLOAT32: fan-out Load×2 + Tanh<D1> + SfpuSub<D0,D1,D0>
    // INP_FLOAT:   Load<WaitNoPop> + Tanh<D0> + DestReuseOp<ELWSUB,SRCB,NoWaitPop>
#ifdef INP_FLOAT32
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
        Tanh<Approx::Exact, Dst::D1>{},
        SfpuSub<Dst::D0, Dst::D1, Dst::D0>{});
#endif
#ifdef INP_FLOAT
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Tanh<Approx::Exact, Dst::D0>{},
        DestReuseOp<
            cb_input,
            EltwiseBinaryType::ELWSUB,
            EltwiseBinaryReuseDestType::DEST_TO_SRCB,
            Dst::D0,
            DestReuseInputPolicy::NoWaitPop>{});
#endif
    eltwise_op<cb_output>(chain, EltwiseTileShape::flat(num_tiles));
}
