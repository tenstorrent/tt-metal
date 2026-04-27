// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// GPT-OSS variant of swiglu: alpha = 1.702, clamp_limit = 7.0
struct SwiGLUConfigGPTOSS
{
    static constexpr float alpha       = 1.702f;
    static constexpr float clamp_limit = 7.0f;
};

// Implements GPT-OSS swiglu activation as a binary SFPU kernel.
// Formula: result = (clamp(up, -L, +L) + 1) * clamp(gate, -inf, +L) * sigmoid(alpha * clamp(gate, -inf, +L))
// where L = Config::clamp_limit and alpha = Config::alpha.
//
// CLAMP IMPLEMENTATION NOTE — uses SFPNONLINEAR RELU to AVOID CC predication.
//
// On Quasar, the SFPGT functional model (Confluence "Tensix SFPU ISA",
// page 1170505767, "SFPGT" subsection; per-instruction page 1612382847)
// gates the CC.Res write on the lane's current LaneEnabled state:
//
//     if (LaneEnabled) {
//         if (Mod1 & SFPGT_MOD1_SET_CC) { LaneFlags = IsVcSmaller; }
//     }
//
// Consequence: chained SFPGT+SFPMOV clamp pairs cannot re-enable lanes the
// previous SFPGT disabled — every subsequent comparison shrinks to the
// already-active subset and lanes outside that subset stay un-clamped.
//
// The Confluence-documented "fix" — inserting SFPENCC(0,0) or SFPENCC(1,2)
// between clamp pairs to set CC.Res=1 — empirically corrupts subsequent
// SFPMOVs on the Quasar simulator (PCC collapses from ~0.999 to ~0.64,
// outputs saturate to the clamp constant on every lane). This is a doc-vs-
// silicon discrepancy that is unresolved as of 2026-04; see
// `codegen/references/common-errors.md` ("Quasar SFPU: Chained SFPGT → SFPMOV
// clamp pairs leak CC predication") for the full investigation.
//
// Therefore we implement the clamps via the predication-free arithmetic
// identity (SFPNONLINEAR RELU is hardware-accelerated on Quasar):
//
//     min(x, +L) = +L - relu(+L - x)
//     max(x, -L) = -L + relu(x + L)
//
// 3 SFPU ops per clamp, 9 ops total for the full clip — same instruction
// count as the (broken) predicated version. No SFPENCC bracketing required.
//
// BF16 packed immediates (top 16 bits of FP32) for the default SwiGLUConfigGPTOSS:
//   +clamp_limit = +7.0f -> 0x40E0 (exact in BF16)
//   -clamp_limit = -7.0f -> 0xC0E0 (exact in BF16)
//   alpha        =  1.702f -> 0x3FDA (quantizes to 1.703125, abs err ~1.1e-3)
template <class Config = SwiGLUConfigGPTOSS>
inline void _calculate_swiglu_(const int iterations, const int gate_offset_idx, const int up_offset_idx, const int out_offset_idx)
{
    // Hoisted compile-time scalar constants — written once per call.
    TTI_SFPLOADI(p_sfpu::LREG4, 0 /*Float16_b*/, 0x40E0); // +clamp_limit
    TTI_SFPLOADI(p_sfpu::LREG5, 0 /*Float16_b*/, 0xC0E0); // -clamp_limit
    TTI_SFPLOADI(p_sfpu::LREG6, 0 /*Float16_b*/, 0x3FDA); // alpha

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // Load gate and up from dest at distinct tile bases.
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, gate_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, up_offset_idx + (d << 1));

        // Clamp gate to max = +clamp_limit:  gate = +L - relu(+L - gate)
        // SFPMAD(va=LCONST_1, vb=LREG0, vc=LREG4, vd=LREG2, mod=NEGATE_VB) → LREG2 = -gate + +L = +L - gate
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG2, 0x1 /*NEGATE_VB*/);
        TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::RELU_MODE);
        // SFPMAD(va=LCONST_1, vb=LREG2, vc=LREG4, vd=LREG0, mod=NEGATE_VB) → LREG0 = -tmp + +L = gate_clamped
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, 0x1 /*NEGATE_VB*/);

        // Clip up to [-L, +L]: do max(up, -L) first, then min(up, +L).
        // max(up, -L): up = -L + relu(up - (-L)) = -L + relu(up + L)
        // SFPMAD(LCONST_1, LREG1, LREG4, LREG2, 0) → LREG2 = up + +L
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::RELU_MODE);
        // SFPMAD(LCONST_1, LREG2, LREG5, LREG1, 0) → LREG1 = tmp + (-L) = relu(up+L) - L
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG1, 0);

        // min(up, +L): up = +L - relu(+L - up)
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, 0x1 /*NEGATE_VB*/);
        TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::RELU_MODE);
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG1, 0x1 /*NEGATE_VB*/);

        // up_plus1 = up + 1.0 -> LREG2 = 1.0 * LREG1 + LCONST_1
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);

        // alpha_gate = alpha * gate -> LREG3 = LREG6 * LREG0 + 0
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

        // sig = sigmoid(alpha_gate), in-place on LREG3; LREG7 is scratch work reg.
        _calculate_sigmoid_regs_(p_sfpu::LREG3, p_sfpu::LREG7, p_sfpu::LREG3);

        // glu = gate * sig -> LREG7 = LREG0 * LREG3 + 0
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);

        // result = up_plus1 * glu -> LREG0 (reused) = LREG2 * LREG7 + 0
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Store result to dest at out_offset_idx + (d << 1)
        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
