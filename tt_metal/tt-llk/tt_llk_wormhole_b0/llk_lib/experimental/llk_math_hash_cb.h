// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wormhole B0 SFPU-backed CB hash for DEBUG_CB_HASH.
//
// Implements the same 23-bit FNV-like ("FNV23") lanewise multiplicative hash as the
// Blackhole variant, using WH-compatible SFPU instructions. WH lacks SFPMUL24, so the
// 23-bit multiply is decomposed via shift-and-add of the prime constant:
//   FNV23_PRIME = 0x193 = 1 + 2 + 16 + 128 + 256
//
// Per-lane per-word:  h_lane = (h_lane ^ w) * 0x000193  (truncated to 23 bits)
// Init:               h_lane = 0x1C9DC5                 (= 0x811C9DC5 & 0x7FFFFF)
// Final reduction:    h = XOR of all 32 lanes            (single u32, 23 bits wide)
//
// The hash is bit-identical to the BH SFPU variant: WH SFPLOAD INT32 applies the same
// UnshuffleFP32 permutation as BH (verified against WormholeB0 SFPLOAD.md), so the
// Python golden in test_hash_cb.py applies to both architectures.
//
// WH ISA notes (verified against WormholeB0 ISA docs):
//   SFPXOR (0, VC, VD, 0)    : VD ^= VC                    (VD < 8 required)
//   SFPAND (0, VC, VD, 0)    : VD &= VC                    (VD < 8 required)
//   SFPIADD(0, VC, VD, 4)    : VD += VC, no flag side-effect (Mod1 = CC_NONE|LREG_DST = 4)
//   SFPSHFT(Imm,0, VD, 1)    : VD <<= Imm  in-place        (Mod1 = ARG_IMM = 1)
//   SFPMOV (0, VC, VD, 0)    : VD = VC
//   SFPSHFT2(0,H,TMP,4)      : SHFLSHR1 within each 8-lane group; VD<8 required.
//     Scheduling hazard: SFPNOP required before reading VD on the next cycle.
//     Hardware bug: lane 0 of each group reads a stale vc0 instead of 0.
//     Workaround: SHFLROR1(LCONST_0, LCONST_0) before the loop zeroes vc0 (VD=9<12
//     satisfies the vc0-update condition; write is suppressed since VD=9>=8).
//   SFPSHFT2(0,0,0,1)         : CHAINED_COPY4 — LReg[3][i] = LReg[0][i+8] (i<24).
//     No scheduling constraint for mode 1.
//
// STATUS: ISA-verified draft. Not yet hardware-validated on WH.

#pragma once

#include <cstdint>

#include "ckernel_debug.h"  // dbg_get_array_row(dbg_array_id::DEST, ...)
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "sfpi.h"

using namespace ckernel;

namespace ckernel::sfpu
{

static constexpr std::uint32_t FNV23_INIT  = 0x1C9DC5u;
static constexpr std::uint32_t FNV23_PRIME = 0x000193u;
static constexpr std::uint32_t MASK23      = 0x7FFFFFu;

// SFPIADD Mod1: LREG_DST (reg+reg, Mod1=0) | CC_NONE (no flag update, Mod1=4) = 4
static constexpr int SFPIADD_MOD_NOFLAG = 4;

// DEST row stride in 16B-unit steps: 2 per row of 32 INT32 elements (matches BH).
static constexpr int DEST_ROW_STRIDE_INT32 = 2;

// LReg assignments. Must keep working registers < 8 (SFPSHFT/SFPXOR/SFPAND/SFPIADD VD constraint).
// LReg[5,6]: save-slots for LREG_H across CHAINED_COPY4 (which clobbers LRegs 0-3).
// LReg[7]:   reserved for indirect addressing — do not use as VD.
// LReg[9]:   hardware constant 0 on WH — used as vc0-zero source and SFPMOV source.
static constexpr int LREG_W    = 0; // word from Dst
static constexpr int LREG_H    = 1; // per-lane FNV23 accumulator
static constexpr int LREG_TMP  = 2; // multiply product accumulator / reduce scratch
static constexpr int LREG_TMP2 = 3; // per-shift-term scratch
static constexpr int LREG_MASK = 4; // 0x7FFFFF constant

inline void _llk_math_hash_cb_init_()
{
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_LOWER, FNV23_INIT & 0xFFFFu);
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_INIT >> 16) & 0xFFFFu);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_LOWER, MASK23 & 0xFFFFu);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_UPPER, (MASK23 >> 16) & 0xFFFFu);
}

inline void _llk_math_hash_cb_tile_(std::uint32_t dst_tile_idx)
{
    const int tile_base = int(dst_tile_idx) * 32 * DEST_ROW_STRIDE_INT32;

#pragma GCC unroll 0
    for (int row = 0; row < 32; ++row)
    {
        const int offset = tile_base + row * DEST_ROW_STRIDE_INT32;

        // Load row → 32 u32 values, one per lane, with UnshuffleFP32.
        TT_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_7, offset);

        // h ^= w
        TTI_SFPXOR(0, LREG_W, LREG_H, 0);

        // h = (h * FNV23_PRIME) & 0x7FFFFF  via shift-and-add.
        // FNV23_PRIME = 0x193 = 1 + 2 + 16 + 128 + 256
        TTI_SFPMOV(0, LREG_H, LREG_TMP, 0); // TMP  = H × 1

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TT_SFPSHFT(1, 0, LREG_TMP2, 1);                          // TMP2 = H << 1
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×3)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TT_SFPSHFT(4, 0, LREG_TMP2, 1);                          // TMP2 = H << 4
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×19)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TT_SFPSHFT(7, 0, LREG_TMP2, 1);                          // TMP2 = H << 7
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×147)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TT_SFPSHFT(8, 0, LREG_TMP2, 1);                          // TMP2 = H << 8
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×403 = ×0x193)

        TTI_SFPAND(0, LREG_MASK, LREG_TMP, 0); // TMP &= 0x7FFFFF
        TTI_SFPMOV(0, LREG_TMP, LREG_H, 0);    // H = TMP
    }
}

// Reduce 32 lanes → 1, drain SFPU, read the hash out of DEST via the debug
// array path, and publish it via a single u32 write to L1 followed by a
// ready-flag write. Caller (hash_cb_sfpu) supplies both L1 addresses.
//
// Reduction proceeds in two phases:
//   Phase 1 — 8-lane intra-group XOR reduce (4 × SHFLSHR1 + XOR):
//     After 4 rounds: lane 0 of groups 0,1,2,3 (at LREG_H lanes 0,8,16,24)
//     carries G0..G3.
//   Phase 2 — cross-group XOR via SFPSHFT2 CHAINED_COPY4 (mode 1):
//     Three +8 shifts fold G1, G2, G3 into lane 0. CHAINED_COPY4 clobbers
//     LRegs 0-3, so LReg[5] accumulates and LReg[6] backs up H for round 1.
inline void _llk_math_hash_cb_finish_to_l1_(std::uint32_t l1_hash_addr, std::uint32_t l1_ready_addr)
{
    // Phase 1: zero vc0 (WH SHFLSHR1 hardware bug workaround).
    // SHFLROR1 with VD=LCONST_0 (index 9): VD=9<12 updates vc0=LReg[9]=0; write suppressed (9>=8).
    TTI_SFPSHFT2(0, p_sfpu::LCONST_0, p_sfpu::LCONST_0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    for (int r = 0; r < 4; ++r)
    {
        TTI_SFPSHFT2(0, LREG_H, LREG_TMP, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
        TTI_SFPNOP; // required: VD (LREG_TMP) read hazard after SHFLSHR1
        TTI_SFPXOR(0, LREG_TMP, LREG_H, 0);
    }
    // LREG_H: G0 at lane 0, G1 at lane 8, G2 at lane 16, G3 at lane 24.

    // Phase 2: save LREG_H before CHAINED_COPY4 clobbers LRegs 0-3.
    TTI_SFPMOV(0, LREG_H, 5, 0); // LReg[5] = H  (cross-group XOR accumulator)
    TTI_SFPMOV(0, LREG_H, 6, 0); // LReg[6] = H  (source for first +8 shift)

    // Round 1 (+8): LReg[3] = H[lane+8]  → lane 0 of LReg[3] = G1.
    TTI_SFPMOV(0, 6, 0, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 2, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 3, 0);
    TTI_SFPSHFT2(0, 0, 0, sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4);
    TTI_SFPXOR(0, 3, 5, 0); // LReg[5][0] = G0^G1

    // Round 2 (+16): chain from LReg[3] (= H[lane+8]) to get H[lane+16].
    TTI_SFPMOV(0, 3, 0, 0); // LReg[0] = H[lane+8]
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 2, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 3, 0);
    TTI_SFPSHFT2(0, 0, 0, sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4); // LReg[3] = H[lane+16]
    TTI_SFPXOR(0, 3, 5, 0);                                          // LReg[5][0] = G0^G1^G2

    // Round 3 (+24): chain from LReg[3] (= H[lane+16]) to get H[lane+24].
    TTI_SFPMOV(0, 3, 0, 0); // LReg[0] = H[lane+16]
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 2, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, 3, 0);
    TTI_SFPSHFT2(0, 0, 0, sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4); // LReg[3] = H[lane+24]
    TTI_SFPXOR(0, 3, 5, 0);                                          // LReg[5][0] = G0^G1^G2^G3

    // Stash lane 0 of LReg[5] in DEST[0][row 0] so MATH can read it.
    TTI_SFPMOV(0, 5, LREG_H, 0);
    TT_SFPSTORE(LREG_H, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

    // Drain SFPU + DEST writes before reading DEST from scalar code. Without
    // this, dbg_get_array_row can race the pending DEST write and return a
    // stale value.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

    // Read DEST row 0 via debug array path (8 dwords); lane 0 is rd_data[0].
    std::uint32_t rd_data[8];
    dbg_get_array_row(dbg_array_id::DEST, /*row_addr=*/0, rd_data);
    const std::uint32_t h = rd_data[0];

    // Publish to L1: hash first, then ready flag. Two volatile writes to
    // distinct addresses are not reorderable w.r.t. each other; UNPACK's
    // poll (with invalidate_l1_cache()) observes hash before ready.
    *reinterpret_cast<volatile std::uint32_t*>(l1_hash_addr)  = h;
    *reinterpret_cast<volatile std::uint32_t*>(l1_ready_addr) = 1u;
}

} // namespace ckernel::sfpu
