// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mixed-prev reconfig fallback: at the BinaryFpu, srca has a prev (from CopyTile), srcb is first-emit.
//
// CopyTile(CbA->D0) -> BinaryFpu(CbB,CbC->D1) -> AddBinary(D0+D1->D0) -> PackTile(CbOut). At the
// BinaryFpu, srca rotates CbA->CbB with prev set (2-arg _with_dt srca reconfig) while srcb is
// first-emit CbC (1-arg srcb reconfig). Every element's result feeds the output — net =
// CbA + (CbB + CbC) — so CbA is load-bearing and a botched srca reconfig (CbA->CbB) shows up as a
// PCC drop. CbA/CbB/CbC carry distinct dtypes, so the reconfig actually fires (not elided).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(total_tiles),
        CopyTile<cb_a, Dst::D0>{},
        BinaryFpu<
            cb_b,
            cb_c,
            BinaryFpuOp::Add,
            BroadcastDim::None,
            InputLifecycle::Streaming,
            InputLifecycle::Streaming,
            BinaryDataFormatReconfig::Input,
            Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{});
}
