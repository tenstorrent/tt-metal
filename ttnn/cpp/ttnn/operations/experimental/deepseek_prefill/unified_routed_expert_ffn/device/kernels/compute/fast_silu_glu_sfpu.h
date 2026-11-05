// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

// silu(gate) * up as one binary SFPU op over gate and up already in DEST, for the routed-expert
// kernels' plain-SiLU epilogue. One header because the hybrid op compiles the fused and unified
// kernels into a single TU, so each defining its own copy would collide.
//
// sigmoid(g) = 0.5 + 0.5 * tanh(g / 2), with tanh from the 6-entry FP16 SFPLUTFP32 table of
// tanh_init<true> (ckernel_sfpu_tanh.h): max |sigmoid error| 0.0094, saturating to 0/1 at |g| >= 6.
// Op PCC matches the exact SiLU: bfp4 weights and the bf8 output dominate the error.
//
// The init leaves the table in LReg0-2/4-6 and 0.5 in vConstFloatPrgm0, and every tile call relies
// on both, so a kernel using this must run no other SFPU op between the init and its last call.
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_fast_silu_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
    // Slopes packed hi/lo in LReg0/1/2, intercepts in LReg4/5/6 (loaded by fast_silu_glu_init).
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // The six table registers leave two LRegs, so up is loaded only once gate is consumed,
        // and both 0.5 constants come from vConstFloatPrgm0 instead of a loaded register.
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat t = sfpi::lut(gate * sfpi::vConstFloatPrgm0, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Retain);
        t = gate * (t + sfpi::vConstFloatPrgm0);
        // No explicit bf16 rounding: SFPSTORE truncates and the pack to bf8_b rounds again.
        // Rounding first costs ~1 us at ISL 1024 and moves PCC by < 1e-5.
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = t * sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::dst_reg++;
    }
    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

inline void fast_silu_glu_init() {
    // Program the constant first: SFPCONFIG writes it from LReg0, which the table then owns.
    sfpi::vConstFloatPrgm0 = 0.5f;
    // tanh_init<true>'s TABLE1 (breakpoints |v| = 0.5, 1, 1.5, 2, 3; max |err| 0.018) with every
    // slope and intercept halved (exact in fp16), so the LUT returns 0.5 * tanh(v) directly.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.96191406f * 0.5f, 0.57617188f * 0.5f);
    sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.0f, 0.192871094f * 0.5f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.28710938f * 0.5f, 0.0964355469f * 0.5f);
    sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(0.48193359f * 0.5f, 0.76806641f * 0.5f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.0390625f * 0.5f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(0.8828125f * 0.5f, 0.5f);
}
}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace ckernel {
ALWI void fast_silu_glu_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, ckernel::sfpu::fast_silu_glu_init))); }

ALWI void fast_silu_glu_tile(uint32_t gate, uint32_t up, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_fast_silu_glu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */),
        gate,
        up,
        out,
        ckernel::VectorMode::RC)));
}
}  // namespace ckernel
