// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    const uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    // GELU backward: grad_in = grad_out * GELU'(x), tanh-approximation variant.
    //
    // Pre-load: D0 = grad_out (cb_grad_out, WaitAndPop),
    //           D1 = x (cb_input, WaitNoPop — 1st fan-out copy),
    //           D2 = x (cb_input, WaitNoPop — 2nd fan-out copy),
    //           D8 = x (cb_input, NoWaitPop — 3rd fan-out copy, pops tile).
    // Slot D8 requires SyncFull+fp16 mode (16 DEST slots).
    //
    // Both CBs must have the same data format (standard for backward kernels).
    constexpr float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f;
    constexpr float kKappa = 0.044715f;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    auto chain = sfpu_chain(
        Load<cb_grad_out, Dst::D0>{},                      // D0 = grad_out
        Load<cb_input, Dst::D1, LoadPolicy::WaitNoPop>{},  // D1 = x
        Load<cb_input, Dst::D2, LoadPolicy::WaitNoPop>{},  // D2 = x
        Load<cb_input, Dst::D8, LoadPolicy::NoWaitPop>{},  // D8 = x (pops tile)
        // x^3 in D1
        Square<Dst::D1>{},
        SfpuMul<Dst::D1, Dst::D2, Dst::D1>{},
        // kKappa * x^3 in D1
        FillTile<Dst::D3>{kKappa},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},
        // x + kKappa*x^3 in D1 (D2=x)
        SfpuAdd<Dst::D1, Dst::D2, Dst::D1>{},
        // kBeta * (x + kKappa*x^3) in D1
        FillTile<Dst::D3>{kBeta},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},
        // tanh in D1, save to D4
        Tanh<Approx::Exact, Dst::D1>{},
        CopyDest<Dst::D1, Dst::D4>{},
        // CDF: 0.5*(1+tanh) in D1
        FillTile<Dst::D3>{1.0f},
        SfpuAdd<Dst::D1, Dst::D3, Dst::D1>{},
        FillTile<Dst::D3>{0.5f},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},
        // 1 - tanh^2 in D4
        Square<Dst::D4>{},
        FillTile<Dst::D3>{1.0f},
        SfpuSub<Dst::D3, Dst::D4, Dst::D3>{},
        CopyDest<Dst::D3, Dst::D4>{},
        // 1 + 0.134145*x^2 in D2
        FillTile<Dst::D3>{kKappa * 3.0f},
        Square<Dst::D2>{},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},
        FillTile<Dst::D3>{1.0f},
        SfpuAdd<Dst::D2, Dst::D3, Dst::D2>{},
        // PDF: kBeta/2 * (1+0.134145*x^2) * (1-tanh^2) in D2
        SfpuMul<Dst::D2, Dst::D4, Dst::D2>{},
        FillTile<Dst::D3>{kBeta * 0.5f},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},
        // x * pdf in D2 (x from D8)
        CopyDest<Dst::D8, Dst::D3>{},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},
        // grad_out * (CDF + x*PDF) in D0
        SfpuAdd<Dst::D1, Dst::D2, Dst::D1>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        eltwise_op<cb_grad_in, Dst::D0, EltwiseOutputPolicy::Bulk>(chain, EltwiseTileShape::flat(per_core_block_size));
    }
}
