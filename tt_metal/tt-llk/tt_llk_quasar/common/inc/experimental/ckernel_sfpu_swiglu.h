// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Implements the GPT-OSS variant of SwiGLU as a binary SFPU kernel:
//
//     result = (clip(up, -L, +L) + 1) * min(gate, +L) * sigmoid(alpha * min(gate, +L))
//
// where L = clamp_limit = 7.0f and alpha = 1.702f. These constants are loaded
// once per SFPU section by `_init_swiglu_()` (3 BF16 immediates into LREG4/5/6)
// and reused across every face the caller iterates.
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
// outputs saturate to the clamp constant on every lane). Doc-vs-silicon
// discrepancy unresolved as of 2026-04.
//
// We therefore implement the clamps via the predication-free arithmetic
// identities below (SFPNONLINEAR RELU is hardware-accelerated on Quasar):
//
//     min(x, +L)        = +L - relu(+L - x)                  (3 ops)
//     clip(x, -L, +L)   = +L - relu(+2L - relu(x + L))       (5 ops, nested)
//
// Verification of the nested-relu clip:
//     x ≥ +L : relu(x+L) = x+L > 2L  → outer relu = 0       → result = +L
//     x ≤ -L : relu(x+L) = 0          → outer relu = 2L     → result = +L - 2L = -L
//     |x| < L: relu(x+L) = x+L        → outer relu = L - x  → result = +L - (L-x) = x

// Loads the 3 hoisted constants into LREG4/5/6. Call exactly once per SFPU
// section (after `_llk_math_eltwise_sfpu_init_` and before the per-face
// `_calculate_swiglu_` loop). The values persist for the whole section.
//
// `+L` and `+2L` are loaded as BF16 (1 SFPLOADI each) since 7.0f and 14.0f
// are exactly representable in BF16. `alpha = 1.702f` is loaded as full FP32
// via the 2-step UPPER/LOWER pattern (mod0=0x8, mod0=0xA) — the BF16 form
// would quantize to 1.703125 (abs err ~1.1e-3) which then gets amplified by
// the (up+1)*gate*sig multiplication chain. The extra SFPLOADI here is
// amortized across every face in the SFPU section. Same pattern as
// `_init_gelu_` in ckernel_sfpu_gelu.h.
inline void _init_swiglu_()
{
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x40E0); // +clamp_limit   =  +7.0f  (BF16 exact)
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x4160); // +2*clamp_limit = +14.0f  (BF16 exact, used by nested-relu clip)
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x3FD9);  // alpha = 1.702f, FP32 = 0x3FD9DB23 — high half
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0xDB23);  // alpha low half — combined: exact FP32 1.702f
}

// Per-face inner loop: reads `ITERATIONS` (= TEST_FACE_R_DIM / SFP_ROWS)
// 32-datum row-pairs of (gate, up), writes the swiglu output back to Dest.
//
// The body interleaves independent ops from the gate min-clamp, up clip, and
// sigmoid identities so each dependent SFPMAD/SFPNONLINEAR has unrelated work
// between it and its consumer. Same instruction count as the serial form;
// no semantic change. Sigmoid is inlined (rather than calling
// `_calculate_sigmoid_regs_`) so the up_plus1 SFPADD can hide in the multi-
// cycle EXP latency.
//
// LREG layout (lifetimes split across stages — annotations below trace each):
//   LREG0     : gate (in)     → gate_clamped → final result (before store)
//   LREG1     : up (in)       → up_clipped
//   LREG2     : up scratch    → up_plus1
//   LREG3     : gate scratch  → alpha_gate → exp(-alpha_gate) → sigmoid (in-place)
//   LREG4     : +L            (loaded by _init_swiglu_)
//   LREG5     : +2L           (loaded by _init_swiglu_)
//   LREG6     : alpha         (loaded by _init_swiglu_)
//   LREG7     : -alpha_gate   → exp+1 → glu = gate_clamped * sigmoid
template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_swiglu_(const int gate_offset_idx, const int up_offset_idx, const int out_offset_idx)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, gate_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, up_offset_idx + (d << 1));

        // ---- Interleaved gate-clamp (LREG3 scratch) / up-clip inner relu (LREG2 scratch) ----
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG3, 0x1 /*NEGATE_VB*/); // [Gate] LREG3 = +L - gate
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, 0);                 // [Up]   LREG2 = up + L          (hides LREG3 latency)
        TTI_SFPNONLINEAR(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpnonlinear::RELU_MODE);                    // [Gate] LREG3 = relu(+L - gate) (hides LREG2 latency)
        TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::RELU_MODE);                    // [Up]   LREG2 = relu(up + L)    (hides LREG3 latency)

        // ---- Gate-clamp finish / up-clip outer sub (interleaved) ----
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG0, 0x1 /*NEGATE_VB*/); // [Gate] LREG0 = +L - relu = gate_clamped
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0x1 /*NEGATE_VB*/); // [Up]   LREG2 = +2L - relu

        // ---- alpha_gate / up-clip outer relu (interleaved) ----
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0); // [Gate] LREG3 = alpha * gate_clamped
        TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::RELU_MODE);    // [Up]   LREG2 = relu(+2L - relu)

        // ---- Sigmoid start / up-clip finish (interleaved) ----
        TTI_SFPMOV(p_sfpu::LREG3, p_sfpu::LREG7, 1);                                                  // [Sig]  LREG7 = -alpha_gate
        TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG1, 0x1 /*NEGATE_VB*/); // [Up]   LREG1 = +L - relu = up_clipped

        // ---- Sigmoid exp / up_plus1 (interleaved, hides multi-cycle EXP latency) ----
        TTI_SFPNONLINEAR(p_sfpu::LREG7, p_sfpu::LREG3, p_sfpnonlinear::EXP_MODE);        // [Sig]  LREG3 = exp(-alpha_gate)
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0); // [Up]   LREG2 = up_clipped + 1   (hides EXP latency)

        // ---- Sigmoid tail — serial, no independent work left to interleave ----
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, 0); // LREG7 = exp + 1
        TTI_SFPNONLINEAR(p_sfpu::LREG7, p_sfpu::LREG3, p_sfpnonlinear::RECIP_MODE);      // LREG3 = 1/(exp+1) = sigmoid

        // ---- Final multiply chain (serial) ----
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0); // LREG7 = gate_clamped * sigmoid
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // LREG0 = up_plus1 * glu

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
