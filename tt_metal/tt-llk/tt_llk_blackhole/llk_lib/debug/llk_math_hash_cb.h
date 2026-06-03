// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Blackhole SFPU-backed CB hash for DEBUG_CB_HASH.
//
// Computes a 23-bit FNV-like ("FNV23") lanewise multiplicative hash, then writes
// the result back into DEST so the standard packer can move it to L1. Blackhole
// has SFPMUL24 (a 23b × 23b → 23b multiply), so the per-step multiply is a single
// instruction (vs the shift-and-add decomposition on Wormhole).
//
//   Per-lane per-word:  h_lane = (h_lane ^ w) * 0x000193  (truncated to 23 bits)
//   Init:               h_lane = 0x1C9DC5                 (= 0x811C9DC5 & 0x7FFFFF)
//
// Read-back design (see the Wormhole header for the full rationale):
//   - All DEST access uses the proven INT32 SFPU idiom (see ckernel_sfpu_typecast):
//     TTI_SFPLOAD/TTI_SFPSTORE with an addr_mod whose dest auto-increments by 2
//     per step. Hand-rolled ADDR_MOD_7 + explicit offsets do NOT round-trip
//     32-bit DEST datums through the packer. Everything is compile-time, so no
//     TT_/sfpi (dst_reg) forms.
//   - _store_to_dest zeroes the whole tile, then writes the 32 per-lane
//     accumulators into DEST row 0; the host XOR-folds the packed tile and the
//     zeroed rows cancel, leaving XOR(32 accumulators).
//   - The DEST debug-bus read-back (dbg_get_array_row) is intentionally NOT used.

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

static constexpr std::uint32_t FNV23_INIT  = 0x1C9DC5u; // 0x811C9DC5 & 0x7FFFFF
static constexpr std::uint32_t FNV23_PRIME = 0x000193u; // 0x01000193 & 0x7FFFFF

// A 32x32 INT32 tile occupies 32 SFPU rows (32 lanes each). The addr_mod below
// auto-increments the DEST pointer by DEST_ROW_STRIDE per SFPLOAD/SFPSTORE.
static constexpr int DEST_ROWS       = 32;
static constexpr int DEST_ROW_STRIDE = 2; // 32-bit DEST: 64 rows / 32 SFPU rows

// LReg assignments. LReg7 reserved per ISA for indirect-mode encodings; use 0..6.
static constexpr int LREG_W     = 0; // scratch / word load
static constexpr int LREG_PRIME = 1; // FNV23 prime constant
static constexpr int LREG_H     = 2; // per-lane accumulator (persists across tiles)
static constexpr int LREG_ZERO  = 3; // 0 constant, for clearing dest rows

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

    // Load FNV23 init constant into LREG_H (broadcast across all 32 lanes).
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_LOWER, FNV23_INIT & 0xFFFFu);
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_INIT >> 16) & 0xFFFFu);

    // Load FNV23 prime into LREG_PRIME (kept live for the entire probe).
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_LOWER, FNV23_PRIME & 0xFFFFu);
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_PRIME >> 16) & 0xFFFFu);
}

// Fold one DEST tile (already populated by the datacopy) into the 32 per-lane
// accumulators in LREG_H. Walks all 32 SFPU rows via the auto-incrementing addr_mod.
inline void _llk_math_hash_cb_tile_(std::uint32_t /*dst_tile_idx*/)
{
    math::reset_counters(p_setrwc::SET_ABD_F);

#pragma GCC unroll 0
    for (int row = 0; row < DEST_ROWS; ++row)
    {
        // INT32 mode reads 32 bits verbatim (subject to the Dst UnshuffleFP32
        // that applies to every 32b SFPLOAD). ADDR_MOD_3 auto-advances the row.
        TTI_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_3, 0);

        // h ^= w  (per-lane 32b XOR; top 9 bits dropped by next mul)
        TTI_SFPXOR(0, LREG_W, LREG_H, 0);

        // h *= prime (SFPMUL24_MOD1_LOWER masks operands and result to 23b)
        TTI_SFPMUL24(LREG_H, LREG_PRIME, p_sfpu::LCONST_0, LREG_H, sfpi::SFPMUL24_MOD1_LOWER);
    }
}

// Write the per-lane accumulators back into DEST for the packer: zero the whole
// tile, then store the 32 accumulators into row 0. See the Wormhole header for
// why the host-side whole-tile XOR-fold then recovers XOR(32 accumulators).
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
