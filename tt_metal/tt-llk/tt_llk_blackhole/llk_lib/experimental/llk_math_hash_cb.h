// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Experimental SFPU-backed CB hash for DEBUG_CB_HASH.
//
// This is the SFPU counterpart to the plain RISC-V scalar hash that lives in
// tt_metal/hw/ckernels/{arch}/metal/llk_api/debug/llk_hash_cb_api.h. Because
// Blackhole SFPU's integer multiply is SFPMUL24 (actually a 23b x 23b -> 23b
// instruction, not 24b -- see tt-isa-documentation BlackholeA0 SFPMUL24.md),
// a bit-exact 32-bit FNV-1a would require ~5-7 SFPU ops per FNV step plus
// careful UPPER+LOWER combining and handling of the Dst UnshuffleFP32 hazard.
// Rather than pretend we can do exact FNV-1a-32 on this datapath, this variant
// is an honestly-23-bit lanewise multiplicative hash ("FNV23"): the low 23 bits
// of the FNV-32 algorithm, run in parallel across all 32 SFPU lanes, with a
// tree-XOR reduction to lane 0 at the end.
//
//   Per-lane per-word: h_lane = (h_lane ^ w) * 0x000193     (truncated to 23b)
//   Init:              h_lane = 0x1C9DC5                    (= 0x811C9DC5 & 0x7FFFFF)
//   Final reduction:   h = XOR of all 32 lanes              (single u32, 23b wide)
//
// The produced hash is NOT bit-equal to the TRISC scalar variant's output --
// it's a different algorithm. It is still deterministic and is useful as a
// bisection fingerprint so long as callers compare SFPU-hashes to SFPU-hashes.
//
// This file does not run by itself -- it expects the caller to have already
// unpacked input tiles into DEST rows via the normal LLK unpack path. See the
// LLK-API wrapper in tt_metal/hw/ckernels/blackhole/metal/llk_api/debug/
// llk_math_hash_cb_api.h for the orchestrated call sequence.
//
// STATUS: best-effort draft authored without hardware access. The instruction
// sequence below follows the ISA docs but has not been hardware-validated.
// Reviewers with device access should expect to iterate on the exact
// SFPLOAD/SFPSTORE modes, lane-reduction ordering, and DEST addressing before
// merge.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "sfpi.h"

using namespace ckernel;

namespace ckernel::sfpu
{

// Constants for the 23-bit FNV variant. Both fit in SFPMUL24_MOD1_LOWER's 23-bit input window.
static constexpr std::uint32_t FNV23_INIT  = 0x1C9DC5u; // 0x811C9DC5 & 0x7FFFFF
static constexpr std::uint32_t FNV23_PRIME = 0x000193u; // 0x01000193 & 0x7FFFFF

// LReg assignments. LReg7 reserved as per ISA for indirect-mode encodings; use 0..6.
//   LReg0 = scratch / word load
//   LReg1 = scratch / prime constant
//   LReg2 = accumulator (h_lane)
//   LReg3 = reduction scratch
static constexpr int LREG_W     = 0;
static constexpr int LREG_PRIME = 1;
static constexpr int LREG_H     = 2;
static constexpr int LREG_TMP   = 3;

// One DEST row holds 32 lanes x 32b each; for a 32x32 int32 tile we iterate over 32 rows.
// DEST addressing on SFPU is in 16B strides, so row r within tile t lives at:
//   offset = (t * 32 + r) * (32b ? 2 : 1)
// where the "*2" is the 32b-per-element stride expressed as 16B chunks (2 x 16B per row).
static constexpr int DEST_ROW_STRIDE_INT32 = 2; // 16B units per DEST row in INT32 mode

// Initialise the SFPU accumulator. Call once before any _llk_math_hash_cb_tile_ call.
inline void _llk_math_hash_cb_init_()
{
    // Load FNV23 init constant into LREG_H (broadcast across all 32 lanes).
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_LOWER, FNV23_INIT & 0xFFFFu);
    TTI_SFPLOADI(LREG_H, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_INIT >> 16) & 0xFFFFu);

    // Load FNV23 prime into LREG_PRIME (kept live for the entire probe).
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_LOWER, FNV23_PRIME & 0xFFFFu);
    TTI_SFPLOADI(LREG_PRIME, sfpi::SFPLOADI_MOD0_UPPER, (FNV23_PRIME >> 16) & 0xFFFFu);
}

// Accumulate one DEST tile into the running per-lane hash. The tile must already
// be unpacked to DEST at `dst_tile_idx` in INT32 format (32x32 u32 values).
//
// For each of 32 rows in the tile:
//   LREG_W = SFPLOAD<INT32>(dst offset for this row)   -- 32 u32s, one per lane
//   LREG_H = LREG_H XOR LREG_W                         -- per-lane, 32b
//   LREG_H = SFPMUL24_LOWER(LREG_H, LREG_PRIME)        -- per-lane, masks to 23b
inline void _llk_math_hash_cb_tile_(std::uint32_t dst_tile_idx)
{
    const int tile_base = int(dst_tile_idx) * 32 * DEST_ROW_STRIDE_INT32;

#pragma GCC unroll 0
    for (int row = 0; row < 32; ++row)
    {
        const int offset = tile_base + row * DEST_ROW_STRIDE_INT32;

        // Load 32 u32 words (one per lane) from this DEST row. INT32 mode reads
        // all 32 bits verbatim into LReg, subject to the Dst UnshuffleFP32 that
        // applies to every 32b SFPLOAD -- the Python golden must match this shuffle.
        TT_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_7, offset);

        // h ^= w  (per-lane, full 32b XOR; top 9 bits will be dropped by next mul)
        TTI_SFPXOR(LREG_H, LREG_W, 0, 0);

        // h *= prime  (SFPMUL24_MOD1_LOWER masks both operands and result to 23 bits)
        TTI_SFPMUL24(LREG_H, LREG_PRIME, p_sfpu::LCONST_0, LREG_H, sfpi::SFPMUL24_MOD1_LOWER);
    }
}

// Collapse the 32 per-lane accumulators in LREG_H down to a single u32 in lane 0
// via tree XOR. Writes the final hash to DEST at (dst_tile_idx, row 0) for the
// packer to emit.
//
// TODO: hardware-validate the SFPSHFT2 lane-shift semantics and ordering.
// The SFPSHFT2 intrinsic shifts lane contents; doc (BlackholeA0 SFPSHFT2.md)
// specifies mod=4 for "shift left by 1 lane". For a standard pairwise XOR
// tree reduction down 32 -> 1 we want shifts by {16, 8, 4, 2, 1} lanes.
inline void _llk_math_hash_cb_reduce_and_store_(std::uint32_t dst_tile_idx)
{
    // Tree-XOR reduction across 32 lanes into lane 0.
    // After each stage lane 0 carries the XOR of the lanes that have been folded in.
    for (int stage = 4; stage >= 0; --stage)
    {
        const int shift_lanes = 1 << stage; // 16, 8, 4, 2, 1
        (void)shift_lanes;
        // LREG_TMP = LREG_H shifted left by shift_lanes lanes
        TTI_SFPMOV(0, LREG_H, LREG_TMP, 0);
        // NOTE: the per-lane-count encoding for SFPSHFT2 is not 1:1 with the shift
        // count -- the ISA uses mod=4 to indicate "shift left by some fixed amount
        // configured elsewhere". A production reduction will likely unroll the 5
        // stages explicitly rather than looping, to match the TTI immediate form
        // used by horizontal_reduce_max in ckernel_sfpu_reduce.h.
        TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, 4);
        TTI_SFPXOR(LREG_H, LREG_TMP, 0, 0);
    }

    // Store the reduced hash from lane 0 back to DEST row 0 of the target tile
    // so that the packer can move it to the cb_hash output CB.
    const int offset = int(dst_tile_idx) * 32 * DEST_ROW_STRIDE_INT32;
    TT_SFPSTORE(LREG_H, InstrModLoadStore::INT32, ADDR_MOD_7, offset);
}

} // namespace ckernel::sfpu
