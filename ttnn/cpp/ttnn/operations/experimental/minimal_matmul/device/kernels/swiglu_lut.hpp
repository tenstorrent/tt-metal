// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#endif

/**
 * Fused SwiGLU epilogue pass for the minimal_matmul and all_gather_minimal_matmul_async SwiGLU kernels, compiled in
 * only with -DSWIGLU_LUT_SILU (see kSwigluLutSilu below):
 *
 *     out = silu(gate) * up = gate * sigmoid(gate) * up
 *
 * in a single SFPU pass over the gate tile. The sigmoid is one SFPLUTFP32 instruction: a sign-symmetric
 * 6-segment piecewise-linear table with fp16 coefficients and cutoffs at |x| = 0.5, 1, 1.5, 2 and 4
 * (sfpi LutMode::Fp16x6_HWM4). The table approximates sigmoid(x) - 0.5 = sign(x) * (s_k * |x| + i_k), which is
 * odd in x, so the hardware's sign-retaining mode reconstructs the negative half. Coefficients were fitted per
 * segment (least squares, reweighted towards minimax) on [0, 4]; the last segment is the constant 0.5, i.e.
 * sigmoid = 1. Max |error| of the sigmoid is 0.018 at |x| = 4 and about 1e-3 for |x| < 2.
 *
 * Compared with the exp + Newton-reciprocal silu, this replaces ~35-40 SFPU instructions per vector with one, and
 * fusing the `up` multiply removes the separate mul_binary pass and its re-init. The exact path (silu_tile +
 * mul_binary_tile) remains the default build.
 *
 * DST layout: the gate tile at `gate_idst`, the up tile at `gate_idst + 1`; the result overwrites the gate tile.
 * Consecutive DST tiles are 32 sfpi rows apart whatever the dest accumulator format (the same constant the
 * eltwise binary SFPU LLK uses), so the pass works with fp32 dest on or off.
 */

// Opt-in only: the LUT epilogue runs when the kernel is compiled with -DSWIGLU_LUT_SILU (a program-factory define).
// Nothing sets it today, so both SwiGLU kernels keep the exact silu_tile + mul_binary_tile epilogue by default.
#if defined(SWIGLU_LUT_SILU)
constexpr bool kSwigluLutSilu = true;
#else
constexpr bool kSwigluLutSilu = false;
#endif

namespace ckernel {

#ifdef TRISC_MATH
namespace sfpu {

inline void swiglu_lut_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Slope pairs (segments 0|1, 2|3, 4|5) in LReg0-2 and intercept pairs in LReg4-6, the register assignment
    // sfpi::lut<LutMode::Fp16x6_*> reads.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.24492f, 0.21720f);  // |x| in [0, 0.5), [0.5, 1)
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.00049f, 0.01508f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.17303f, 0.12645f);  // [1, 1.5), [1.5, 2)
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(0.05953f, 0.12930f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.05061f, 0.0f);  // [2, 4), [4, inf)
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(0.29043f, 0.5f);
}

template <int ITERATIONS>
inline void calculate_swiglu_lut() {
    constexpr int UP_TILE_OFFSET = 32;  // sfpi rows from a DST tile to the next
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[0];
        sfpi::vFloat sigmoid =
            sfpi::lut<sfpi::LutMode::Fp16x6_HWM4>(gate, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain) + 0.5f;
        sfpi::dst_reg[0] = gate * sigmoid * sfpi::dst_reg[UP_TILE_OFFSET];
        sfpi::dst_reg++;
    }
    // Write the coefficients back so the compiler keeps them pinned in the LRegs across the loop.
    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

}  // namespace sfpu
#endif  // TRISC_MATH

// Programs the SFPU and the sigmoid table for swiglu_lut_tile. Like every SFPU init it must run after any other
// SFPU program (copy/bcast do not count) and before the first swiglu_lut_tile that follows it.
ALWI void swiglu_lut_tile_init() {
    MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::silu>(::ckernel::sfpu::swiglu_lut_init)));
}

// DST[gate_idst] = silu(DST[gate_idst]) * DST[gate_idst + 1], full tile.
ALWI void swiglu_lut_tile(uint32_t gate_idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, false, calculate_swiglu_lut, (8), gate_idst, ::ckernel::VectorMode::RC));
}

}  // namespace ckernel
