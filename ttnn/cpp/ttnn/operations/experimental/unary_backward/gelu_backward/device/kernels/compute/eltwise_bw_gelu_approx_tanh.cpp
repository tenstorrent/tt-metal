// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_grad_out = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_grad_in = static_cast<uint32_t>(tt::CBIndex::c_2);

    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kKappa = 0.044715;
    constexpr float kKappa3 = kKappa * 3.0f;
    constexpr float kHalfBeta = kBeta / 2.0f;

    init_sfpu(cb_grad_out, cb_grad_in);

    // Tanh-approximation GELU backward:
    //   cdf = 0.5 * (1 + tanh(β * (x + κ·x³)))
    //   pdf = 0.5·β · (1 + 3κ·x²) · (1 - tanh²(β · (x + κ·x³)))
    //   grad_in = grad_out * (cdf + x · pdf)
    //
    // Original used DEST slot 8 as a backup of x — out of Dst::D7 range and
    // not needed: CopyDest from D1 preserves x without a second CB load.
    //
    // DEST slot map:
    //   D0 = grad_out (also final result)
    //   D1 = x, then working register for cdf calc
    //   D2 = working register for pdf calc (starts as x, gets squared)
    //   D3 = scratch for FillScalar constants
    //   D4 = tanh backup, then (1 - tanh²)
    //   D5 = x backup (saved before D1 is consumed by cdf work)
    auto chain = sfpu_chain(
        Load<cb_grad_out, Dst::D0, LoadPolicy::WaitUpfrontPopAtEnd>{},
        Load<cb_input, Dst::D1, LoadPolicy::WaitUpfrontPopAtEnd>{},
        CopyDest<Dst::D1, Dst::D2>{},
        CopyDest<Dst::D1, Dst::D5>{},

        // D1 = β · (x + κ·x³)
        Square<Dst::D1>{},
        SfpuMul<Dst::D1, Dst::D2, Dst::D1>{},
        FillScalar<Dst::D3>{kKappa},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},
        SfpuAdd<Dst::D1, Dst::D2, Dst::D1>{},
        FillScalar<Dst::D3>{kBeta},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},

        // D1 = tanh(β · (x + κ·x³));  D4 = copy of D1
        Tanh<Approx::Exact, Dst::D1>{},
        CopyDest<Dst::D1, Dst::D4>{},

        // D1 = 0.5 · (1 + tanh) = cdf
        FillScalar<Dst::D3>{1.0f},
        SfpuAdd<Dst::D1, Dst::D3, Dst::D1>{},
        FillScalar<Dst::D3>{0.5f},
        SfpuMul<Dst::D1, Dst::D3, Dst::D1>{},

        // D4 = 1 - tanh²  (via D3 = 1 - D4 then CopyDest back to D4)
        Square<Dst::D4>{},
        FillScalar<Dst::D3>{1.0f},
        SfpuSub<Dst::D3, Dst::D4, Dst::D3>{},
        CopyDest<Dst::D3, Dst::D4>{},

        // D2 = 1 + 3κ·x²
        FillScalar<Dst::D3>{kKappa3},
        Square<Dst::D2>{},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},
        FillScalar<Dst::D3>{1.0f},
        SfpuAdd<Dst::D2, Dst::D3, Dst::D2>{},

        // D2 = pdf = 0.5·β · (1 + 3κ·x²) · (1 - tanh²)
        SfpuMul<Dst::D2, Dst::D4, Dst::D2>{},
        FillScalar<Dst::D3>{kHalfBeta},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},

        // D2 = x · pdf  (x restored from D5 backup)
        CopyDest<Dst::D5, Dst::D3>{},
        SfpuMul<Dst::D2, Dst::D3, Dst::D2>{},

        // D0 = grad_out · (cdf + x · pdf)
        SfpuAdd<Dst::D1, Dst::D2, Dst::D1>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_grad_in, per_core_block_size);
    }
}
