// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Blackhole SFPU-backed CB hash for DEBUG_CB_HASH.
//
// Computes a 23-bit FNV-like ("FNV23") lanewise multiplicative hash, then writes
// the result back into DEST for the packer to move to L1 (no DEST debug-bus
// read-back: dbg_get_array_row does not round-trip 32-bit DEST datums).
//
//   Per-lane per-word:  h_lane = (h_lane ^ w) * 0x000193  (truncated to 23 bits)
//   Init:               h_lane = 0x1C9DC5                 (= 0x811C9DC5 & 0x7FFFFF)
//
// Blackhole has SFPMUL24 (a 23b × 23b → 23b multiply), so the per-step multiply
// is a single instruction (Wormhole decomposes it into shift-and-add).
//
// DEST addressing:
//   A 32x32 INT32 tile is walked in 32 SFPLOAD/SFPSTORE steps (one per SFPU row
//   of 32 lanes). SFPLOAD/SFPSTORE auto-advance the DEST row counter by the
//   addr_mod's Dst increment, so each step's offset operand is a constant 0 and
//   the whole sequence is compile-time — no TT_ (runtime) or sfpi (dst_reg++)
//   forms. DEST_ROW_STRIDE = 2 is the per-step Dst-row increment for 32-bit DEST
//   (same value the standard INT32 SFPU ops use, e.g. ckernel_sfpu_typecast).
//
//   We use ADDR_MOD_1. Blackhole SFPLOAD/SFPSTORE encode a 3-bit addr_mod field
//   (slots 0..7 are reachable, unlike Wormhole's 2-bit field which reaches only
//   0..3). The A2D datacopy that feeds DEST owns ADDR_MOD_0/2/3, so ADDR_MOD_1 is
//   free; we pick it to mirror the Wormhole path (where it is the only free,
//   reachable slot).
//
// Read-back / fold:
//   _store_to_dest first zeroes the whole tile, then writes the 32 per-lane
//   accumulators into DEST row 0. The packer moves the tile to L1 and a scalar
//   consumer XOR-folds the whole tile. The zeroed datums all read back as the
//   same word (whatever the pack format conversion maps 0 to), and there is an
//   even number of them (31 rows x 32 lanes), so they cancel under XOR — leaving
//   the fold of just the 32 accumulator words. The result is therefore a
//   deterministic, input-sensitive fingerprint, independent of the SFPU lane <->
//   tile-position permutation and of any fixed pack datum conversion.

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

// A 32x32 INT32 tile is 32 SFPU rows (32 lanes each); ADDR_MOD_1 advances the
// DEST row counter by DEST_ROW_STRIDE per SFPLOAD/SFPSTORE. See file header.
static constexpr int DEST_ROWS       = 32;
static constexpr int DEST_ROW_STRIDE = 2;

// LReg assignments. LReg7 reserved per ISA for indirect-mode encodings; use 0..6.
static constexpr int LREG_W     = 0; // scratch / word load
static constexpr int LREG_PRIME = 1; // FNV23 prime constant
static constexpr int LREG_H     = 2; // per-lane accumulator (persists across tiles)
static constexpr int LREG_ZERO  = 3; // 0 constant, for clearing dest rows

/**
 * @brief Seed the per-lane FNV23 accumulators and configure DEST addressing.
 *
 * Configures the SFPU config register and ADDR_MOD_1 (Dst auto-increment) so the
 * tile/store loops can use plain TTI_SFPLOAD/TTI_SFPSTORE(..., ADDR_MOD_1, 0).
 *
 * @pre Call once before any @ref _llk_math_hash_cb_tile_ in this probe.
 * @note Overwrites ADDR_MOD_1, LREG_H and LREG_PRIME.
 */
inline void _llk_math_hash_cb_init_()
{
    sfpu::_init_sfpu_config_reg();
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = DEST_ROW_STRIDE},
    }
        .set(ADDR_MOD_1);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Load FNV23 init constant into LREG_H (broadcast across all 32 lanes).
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_LOWER, FNV23_INIT & 0xFFFFu);
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_INIT >> 16) & 0xFFFFu);

    // Load FNV23 prime into LREG_PRIME (kept live for the entire probe).
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_LOWER, FNV23_PRIME & 0xFFFFu);
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_PRIME >> 16) & 0xFFFFu);
}

/**
 * @brief Fold one INT32 DEST tile into the 32 per-lane FNV23 accumulators.
 *
 * Walks all 32 SFPU rows of the tile via ADDR_MOD_1's auto-incrementing Dst
 * counter, folding each row's 32 lanes into LREG_H.
 *
 * @param dst_tile_idx: Unused — the orchestration always folds DEST slot 0.
 * @pre @ref _llk_math_hash_cb_init_, and the tile's data already in DEST slot 0.
 */
inline void _llk_math_hash_cb_tile_(std::uint32_t /*dst_tile_idx*/)
{
    math::reset_counters(p_setrwc::SET_ABD_F);

#pragma GCC unroll 0
    for (int row = 0; row < DEST_ROWS; ++row)
    {
        // INT32 mode reads 32 bits verbatim (subject to the Dst UnshuffleFP32
        // that applies to every 32b SFPLOAD). ADDR_MOD_1 auto-advances the row.
        TTI_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_1, 0);

        // h ^= w  (per-lane 32b XOR; top 9 bits dropped by next mul)
        TTI_SFPXOR(0, LREG_W, LREG_H, 0);

        // h *= prime (SFPMUL24_MOD1_LOWER masks operands and result to 23b)
        TTI_SFPMUL24(LREG_H, LREG_PRIME, p_sfpu::LCONST_0, LREG_H, sfpi::SFPMUL24_MOD1_LOWER);
    }
}

/**
 * @brief Write the per-lane accumulators back into DEST for the packer.
 *
 * Zeroes the whole tile, then stores the 32 accumulators into DEST row 0, so a
 * whole-tile XOR-fold of the packed result recovers the fold of the 32
 * accumulators (the even count of equal zero words cancels). See file header.
 *
 * @pre @ref _llk_math_hash_cb_tile_ has folded all input tiles.
 * @post Caller packs DEST and the host XOR-folds the result tile.
 * @note Clobbers all of DEST slot 0 and LREG_ZERO.
 */
inline void _llk_math_hash_cb_store_to_dest_()
{
    TTI_SFPLOADI(LREG_ZERO, sfpi::SFPLOADI_MOD0_LOWER, 0u);
    TTI_SFPLOADI(LREG_ZERO, sfpi::SFPLOADI_MOD0_UPPER, 0u);

    // Zero every row of the tile (ADDR_MOD_1 auto-advances the Dst row).
    math::reset_counters(p_setrwc::SET_ABD_F);
#pragma GCC unroll 0
    for (int row = 0; row < DEST_ROWS; ++row)
    {
        TTI_SFPSTORE(LREG_ZERO, InstrModLoadStore::INT32, ADDR_MOD_1, 0);
    }

    // Overwrite row 0 with the 32 per-lane accumulators.
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_SFPSTORE(LREG_H, InstrModLoadStore::INT32, ADDR_MOD_1, 0);

    // Drain the SFPU pipeline so the packer sees the final dest contents.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

} // namespace ckernel::sfpu
