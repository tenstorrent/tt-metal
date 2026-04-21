// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Test: y = x * exp(x) via sfpu_chain fan-out.
//
// Exercises:
// - Load fan-out without CompactLoad: WaitNoPop (first) + NoWaitPop (second).
// - SfpuMul as a chain BinaryOp element operating on two pre-loaded DEST slots.
// - Single-compute-op chain (Exp) path after the num_compute_ops removal —
//   must still produce correct results now that chain.apply is called every tile.

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    init_sfpu(cb_input, cb_output);

    using namespace compute_kernel_lib;
    sfpu_pipeline<
        SfpuBatching::Auto,
        SfpuInputPolicy::WaitAndPopPerTile,
        SfpuOutputPolicy::PerTile,
        SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(
        sfpu_chain(
            Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
            Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{}),
        cb_output,
        num_tiles);
}
}  // namespace NAMESPACE
