// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    init_sfpu(cb_input, cb_output);

    // mish(x) = x * tanh(log1p(exp(x)))
    // Output CB has capacity 2 → use PerTile output policy.
    // use_approx is runtime: dispatch to two statically-typed chains.
    if (use_approx) {
#ifdef INP_FLOAT32
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
            Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
#endif
#ifdef INP_FLOAT
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            DestReuseOp<
                cb_input,
                EltwiseBinaryType::ELWMUL,
                EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                Dst::D0,
                LoadPolicy::WaitAndPop>{});
#endif
        sfpu_pipeline<SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    } else {
#ifdef INP_FLOAT32
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
            Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
#endif
#ifdef INP_FLOAT
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            DestReuseOp<
                cb_input,
                EltwiseBinaryType::ELWMUL,
                EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                Dst::D0,
                LoadPolicy::WaitAndPop>{});
#endif
        sfpu_pipeline<SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    }
}
