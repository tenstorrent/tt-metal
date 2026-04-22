// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    init_sfpu(cb_input, cb_output);

    // hardswish(x) = x * hardsigmoid(x)
    // Output CB has capacity 2 → use PerTile output policy.
#ifdef INP_FLOAT32
    // FP32: SFPU mul between two DEST slots (no FPU binary needed)
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
        Hardsigmoid<Dst::D0>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
#endif
#ifdef INP_FLOAT
    // BFloat16: FPU dest-reuse mul preserves original precision path.
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Hardsigmoid<Dst::D0>{},
        DestReuseOp<
            cb_input,
            EltwiseBinaryType::ELWMUL,
            EltwiseBinaryReuseDestType::DEST_TO_SRCA,
            Dst::D0,
            LoadPolicy::WaitAndPop>{});
#endif

    sfpu_pipeline<SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
