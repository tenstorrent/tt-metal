// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Only numerator DST addresses change. Same SFPU instruction/op order, macros,
// BF16 rounding and full FP32 live residual. Denominator replay stays original.
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
inline void init_sdpa_compensated_block_macros() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    // Templates 0/1 round fixed VC=L0/L3. Override VB with the load's
    // register, leaving VC intact; store that rounded register two instructions
    // later at the load's captured DST address.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 3, 3, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0x1384);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, 0x8, 0x1385);
    TTI_SFPCONFIG(0, 5, 0);
    // Templates 2/3 add the loaded low component to the fixed high component.
    TTI_SFPADD(10, 0, 1, 14, 0);
    TTI_SFPADD(10, 3, 2, 15, 0);
    TTI_SFPCONFIG(0x600, 6, 1);
    TTI_SFPCONFIG(0x700, 7, 1);
    TTI_SFPCONFIG(0xF00, 8, 1);  // BF16 stores; delays count SFPU instructions.

    // Record only: no DST access occurs until a subsequent tile_regs_wait.
    // DST holds (hi0, hi1, lo0, lo1, chunk0, chunk1, correction).
    // L0/L3 retain full sums; L1/L2 hold rounded results; L4/L5 are chunks.
    // Keep macro-load destinations below L4: VDHi also encodes address bit 0.
    TTI_REPLAY(0, 15, 0, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 192);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    TTI_SFPMAD(2, 6, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 64);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 192);
}

}  // namespace ckernel::sfpu
#endif
