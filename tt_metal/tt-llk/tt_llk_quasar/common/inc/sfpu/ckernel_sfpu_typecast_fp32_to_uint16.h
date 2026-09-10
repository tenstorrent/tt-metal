// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
//
// A Float32 source forces 32-bit Dest, so the UInt16 result is a narrow datum living inside a
// 32-bit Dest word. WH/BH reached that half-word by setting debug feature bit 11; Quasar has no
// such bit, so every access below names its format explicitly instead of using sfpmem::DEFAULT,
// which would make HW re-derive the format from ALU_FORMAT_SPEC_REG / ACC_CTRL_SFPU_Fp32. The
// caller must configure the math thread with implied math format disabled for the same reason.
inline void _calculate_typecast_fp32_to_uint16_rows()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, 0); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    // Clamp negatives to 0 before the cast; the unsigned conversion does not saturate them.
    // SFPENCC instr_mod1 bits[1:0] select the CC-enable source: 2 takes it from imm12_math[0],
    // so (1, 2) turns predication on and (0, 2) turns it back off. Mode 1 would only invert the
    // previous enable, which leaves the block's behaviour dependent on incoming CC state.
    // SFPSETCC imm12_math bit 11 selects how src_c is read: it must be set for an FP32 LREG,
    // otherwise the float bits are compared as two's-complement int32.
    TTI_SFPENCC(1, 2);                     // CC_en <= 1, CC_res <= 1
    TTI_SFPSETCC(0x800, p_sfpu::LREG0, 0); // CC_res <= (LREG0 < 0), src read as FP32
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0);     // loads zeros where lreg[0] is negative
    TTI_SFPENCC(0, 2);                     // CC_en <= 0, subsequent lanes all active

    // Same two-step convert as calculate_typecast: fp32 → sign-mag int32 (SFPCAST 0x4 = RNE),
    // then a 16-bit store that names the Dest half-word. Do not use SFP_STOCH_RND FP32_TO_UINT16.
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, 0x4);

    // sfpmem::UINT16 is the unsigned-16 store mode: it narrows the int32 in lreg[1] and places it
    // where the packer reads UInt16 from, the same mode calculate_typecast and
    // _calculate_typecast_fp16b_to_uint16_rows use.
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::UINT16, ADDR_MOD_7, 0, 0); // Store from lreg[1] into dest register
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_typecast_fp32_to_uint16_rows();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
