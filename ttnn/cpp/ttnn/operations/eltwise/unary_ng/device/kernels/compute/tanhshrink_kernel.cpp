// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

    // tanhshrink(x) = x - tanh(x)
    // Output CB has capacity 2 → use PerTile output policy.
#ifdef INP_FLOAT32
    // FP32: load x to D1, tanh(D1), load x to D0, D0 - D1 via SFPU sub
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D1, LoadPolicy::WaitNoPop>{},
        Tanh<Approx::Exact, Dst::D1>{},
        Load<cb_input, Dst::D0, LoadPolicy::NoWaitPop>{},
        SfpuSub<Dst::D0, Dst::D1, Dst::D0>{});
#endif
#ifdef INP_FLOAT
    // BFloat16: FPU dest-reuse sub — DEST_TO_SRCB loads x into SRCB,
    // computes SRCA(tanh(x)) - SRCB(x). Wait: ELWSUB(SRCA, SRCB) = SRCA - SRCB,
    // so result = tanh(x) - x. But raw code intended x - tanh(x), so DEST_TO_SRCB
    // means DST goes to SRCB and CB goes to SRCA: result = cb(x) - dst(tanh(x)) = x - tanh(x). ✓
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Tanh<Approx::Exact, Dst::D0>{},
        DestReuseOp<
            cb_input,
            EltwiseBinaryType::ELWSUB,
            EltwiseBinaryReuseDestType::DEST_TO_SRCB,
            Dst::D0,
            LoadPolicy::WaitAndPop>{});
#endif

    sfpu_pipeline<SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
