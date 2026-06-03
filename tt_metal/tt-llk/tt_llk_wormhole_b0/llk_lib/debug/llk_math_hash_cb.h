// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wormhole B0 SFPU-backed CB hash for DEBUG_CB_HASH.
//
// Computes a 23-bit FNV-like ("FNV23") lanewise multiplicative hash, then writes
// the result back into DEST so the standard packer can move it to L1. WH lacks
// SFPMUL24, so the 23-bit multiply is decomposed via shift-and-add of the prime:
//   FNV23_PRIME = 0x193 = 1 + 2 + 16 + 128 + 256
//
//   Per-lane per-word:  h_lane = (h_lane ^ w) * 0x000193  (truncated to 23 bits)
//   Init:               h_lane = 0x1C9DC5                 (= 0x811C9DC5 & 0x7FFFFF)
//
// Read-back design (the part that makes this robust):
//   - All DEST access uses the proven INT32 SFPU idiom (see ckernel_sfpu_typecast):
//     TTI_SFPLOAD/TTI_SFPSTORE with an addr_mod whose dest auto-increments by 2
//     per step (a 32-bit DEST tile is 64 rows = 32 SFPU rows × stride 2). This is
//     the only pattern that the packer's tile read agrees with; hand-rolled
//     ADDR_MOD_7 + explicit offsets do NOT (they leave half of each 32-bit datum
//     unwritten). Everything is compile-time, so no TT_/sfpi (dst_reg) forms.
//   - _store_to_dest first zeroes the whole tile, then writes the 32 per-lane
//     accumulators into DEST row 0. The packer moves the tile to L1 and the host
//     XOR-folds the whole tile: the 31 zeroed rows contribute an even number of
//     identical words (the packer applies a fixed, value-preserving transform to
//     every datum) which cancel under XOR, leaving XOR(32 accumulators). The
//     result is therefore independent of the SFPU lane <-> tile-position
//     permutation and of the packer's datum transform — only determinism matters,
//     which holds.
//   - The DEST debug-bus read-back (dbg_get_array_row) is intentionally NOT used:
//     it does not round-trip 32-bit DEST datums and has no other call site.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h" // _init_sfpu_config_reg

using namespace ckernel;

namespace ckernel::sfpu
{

static constexpr std::uint32_t FNV23_INIT  = 0x1C9DC5u;
static constexpr std::uint32_t FNV23_PRIME = 0x000193u; // documents the multiply below
static constexpr std::uint32_t MASK23      = 0x7FFFFFu;

// SFPIADD Mod1: LREG_DST (reg+reg, Mod1=0) | CC_NONE (no flag update, Mod1=4) = 4
static constexpr int SFPIADD_MOD_NOFLAG = 4;

// A 32x32 INT32 tile occupies 32 SFPU rows (32 lanes each). The addr_mod below
// auto-increments the DEST pointer by DEST_ROW_STRIDE per SFPLOAD/SFPSTORE.
static constexpr int DEST_ROWS       = 32;
static constexpr int DEST_ROW_STRIDE = 2; // 32-bit DEST: 64 rows / 32 SFPU rows

// LReg assignments (working registers kept < 8 for SFPXOR/SFPAND/SFPIADD/SFPSHFT VD constraint).
static constexpr int LREG_W    = 0; // word loaded from Dst
static constexpr int LREG_H    = 1; // per-lane FNV23 accumulator (persists across tiles)
static constexpr int LREG_TMP  = 2; // multiply product accumulator
static constexpr int LREG_TMP2 = 3; // per-shift-term scratch
static constexpr int LREG_MASK = 4; // 0x7FFFFF constant
static constexpr int LREG_ZERO = 5; // 0 constant, for clearing dest rows

inline void _llk_math_hash_cb_init_()
{
    // Configure the SFPU like the standard eltwise-unary-SFPU init. ADDR_MOD_3's
    // dest auto-increments by DEST_ROW_STRIDE so a plain loop of TTI_SFPLOAD/
    // TTI_SFPSTORE(..., ADDR_MOD_3, 0) walks the 32 SFPU rows of the tile.
    sfpu::_init_sfpu_config_reg();
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = DEST_ROW_STRIDE},
    }
        .set(ADDR_MOD_3);
    math::reset_counters(p_setrwc::SET_ABD_F);

    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_LOWER, FNV23_INIT & 0xFFFFu);
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_INIT >> 16) & 0xFFFFu);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_LOWER, MASK23 & 0xFFFFu);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_UPPER, (MASK23 >> 16) & 0xFFFFu);
}

// Fold one DEST tile (already populated by the datacopy) into the 32 per-lane
// accumulators in LREG_H. Walks all 32 SFPU rows via the auto-incrementing addr_mod.
inline void _llk_math_hash_cb_tile_(std::uint32_t /*dst_tile_idx*/)
{
    math::reset_counters(p_setrwc::SET_ABD_F);

#pragma GCC unroll 0
    for (int row = 0; row < DEST_ROWS; ++row)
    {
        // ADDR_MOD_3 auto-advances the DEST row by DEST_ROW_STRIDE after the load.
        TTI_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_3, 0);

        // h ^= w
        TTI_SFPXOR(0, LREG_W, LREG_H, 0);

        // h = (h * FNV23_PRIME) & 0x7FFFFF  via shift-and-add.
        // FNV23_PRIME = 0x193 = 1 + 2 + 16 + 128 + 256
        TTI_SFPMOV(0, LREG_H, LREG_TMP, 0); // TMP  = H × 1

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TTI_SFPSHFT(1, 0, LREG_TMP2, 1);                         // TMP2 = H << 1
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×3)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TTI_SFPSHFT(4, 0, LREG_TMP2, 1);                         // TMP2 = H << 4
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×19)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TTI_SFPSHFT(7, 0, LREG_TMP2, 1);                         // TMP2 = H << 7
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×147)

        TTI_SFPMOV(0, LREG_H, LREG_TMP2, 0);
        TTI_SFPSHFT(8, 0, LREG_TMP2, 1);                         // TMP2 = H << 8
        TTI_SFPIADD(0, LREG_TMP2, LREG_TMP, SFPIADD_MOD_NOFLAG); // TMP += TMP2 (×403 = ×0x193)

        TTI_SFPAND(0, LREG_MASK, LREG_TMP, 0); // TMP &= 0x7FFFFF
        TTI_SFPMOV(0, LREG_TMP, LREG_H, 0);    // H = TMP
    }
}

// Write the per-lane accumulators back into DEST for the packer: zero the whole
// tile, then store the 32 accumulators into row 0. See file header for why the
// host-side whole-tile XOR-fold then recovers XOR(32 accumulators) exactly.
inline void _llk_math_hash_cb_store_to_dest_()
{
    TTI_SFPLOADI(LREG_ZERO, sfpi::SFPLOADI_MOD0_LOWER, 0u);
    TTI_SFPLOADI(LREG_ZERO, sfpi::SFPLOADI_MOD0_UPPER, 0u);

    // Zero every row of the tile (ADDR_MOD_3 auto-advances the DEST row).
    math::reset_counters(p_setrwc::SET_ABD_F);
#pragma GCC unroll 0
    for (int row = 0; row < DEST_ROWS; ++row)
    {
        TTI_SFPSTORE(LREG_ZERO, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
    }

    // Overwrite row 0 with the 32 per-lane accumulators.
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_SFPSTORE(LREG_H, InstrModLoadStore::INT32, ADDR_MOD_3, 0);

    // Drain the SFPU pipeline so the packer sees the final dest contents.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

} // namespace ckernel::sfpu
