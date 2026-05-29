// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU-backed CB hash for DEBUG_CB_HASH (Blackhole).
//
// SFPU counterpart to the plain RISC-V scalar hash in
// tt_metal/hw/ckernels/{arch}/metal/llk_api/debug/llk_hash_cb_api.h. Blackhole's
// integer multiply is SFPMUL24 — actually a 23b × 23b → 23b op per
// BlackholeA0/SFPMUL24.md — so an exact FNV-1a-32 would need ~5–7 SFPU ops per
// step plus UPPER+LOWER recombination. Instead this is an honestly-23-bit
// lanewise multiplicative hash ("FNV23"): the low 23 bits of FNV-32 run in
// parallel across all 32 SFPU lanes, with a tree-XOR reduction to lane 0.
//
//   Per-lane per-word: h_lane = (h_lane ^ w) * 0x000193    (truncated to 23b)
//   Init:              h_lane = 0x1C9DC5                   (= 0x811C9DC5 & 0x7FFFFF)
//   Final reduction:   h = XOR of all 32 lanes              (single u32, 23b wide)
//
// The hash is NOT bit-equal to the scalar variant — only compare SFPU-hashes
// to SFPU-hashes.
//
// Caller responsibility (see hash_cb_sfpu in api/compute/debug/cb_hash.h):
//   UNPACK puts INT32 tiles into DEST, MATH calls _init/_tile/_finish_to_l1,
//   UNPACK polls and DPRINTs from L1.
//
// STATUS: draft. Inner SFPU loop and DEST-row read-back follow the BH ISA
// docs (SFPMUL24, SFPSHFT2, SFPLOAD/SFPSTORE INT32, dbg_get_array_row) but
// have not been hardware-validated. Expect iteration on SFPSHFT2 lane-shift
// ordering and the SFPU/DEST drain before the debug-bus read on first run.

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

// Constants for the 23-bit FNV variant. Both fit in SFPMUL24_MOD1_LOWER's 23-bit input window.
static constexpr std::uint32_t FNV23_INIT  = 0x1C9DC5u; // 0x811C9DC5 & 0x7FFFFF
static constexpr std::uint32_t FNV23_PRIME = 0x000193u; // 0x01000193 & 0x7FFFFF

// LReg assignments. LReg7 reserved as per ISA for indirect-mode encodings; use 0..6.
static constexpr int LREG_W     = 0; // scratch / word load
static constexpr int LREG_PRIME = 1; // FNV23 prime constant
static constexpr int LREG_H     = 2; // per-lane accumulator
static constexpr int LREG_TMP   = 3; // reduction scratch

// DEST addressing on SFPU is in 16B strides; an INT32 row of 32 elements
// (128B) is 2 × 16B units.
static constexpr int DEST_ROW_STRIDE_INT32 = 2;

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

// Accumulate one DEST tile into the running per-lane hash. The tile must
// already be unpacked to DEST at `dst_tile_idx` in INT32 format (32×32 u32).
//
//   LREG_W = SFPLOAD<INT32>(row offset)             — 32 u32, one per lane
//   LREG_H = LREG_H XOR LREG_W                       — per-lane, 32b
//   LREG_H = SFPMUL24_LOWER(LREG_H, LREG_PRIME)      — per-lane, masks to 23b
inline void _llk_math_hash_cb_tile_(std::uint32_t dst_tile_idx)
{
    const int tile_base = int(dst_tile_idx) * 32 * DEST_ROW_STRIDE_INT32;

#pragma GCC unroll 0
    for (int row = 0; row < 32; ++row)
    {
        const int offset = tile_base + row * DEST_ROW_STRIDE_INT32;

        // INT32 mode reads 32 bits verbatim, subject to the Dst UnshuffleFP32
        // that applies to every 32b SFPLOAD. Any golden comparison must mirror
        // that permutation.
        TT_SFPLOAD(LREG_W, InstrModLoadStore::INT32, ADDR_MOD_7, offset);

        // h ^= w  (per-lane 32b XOR; top 9 bits dropped by next mul)
        TTI_SFPXOR(0, LREG_W, LREG_H, 0);

        // h *= prime (SFPMUL24_MOD1_LOWER masks operands and result to 23b)
        TTI_SFPMUL24(LREG_H, LREG_PRIME, p_sfpu::LCONST_0, LREG_H, sfpi::SFPMUL24_MOD1_LOWER);
    }
}

// Reduce 32 lanes → 1, drain SFPU, read the hash out of DEST via the debug
// array path, and publish it via a single u32 write to L1 followed by a
// ready-flag write. Caller (hash_cb_sfpu) supplies both L1 addresses.
//
// dst_tile_idx is implicitly 0 — the orchestration in cb_hash.h always
// accumulates into DEST slot 0, and the SFPSTORE here writes row 0 of that
// slot, so the dbg_get_array_row call reads from the same row.
inline void _llk_math_hash_cb_finish_to_l1_(std::uint32_t l1_hash_addr, std::uint32_t l1_ready_addr)
{
    // ---- Tree-XOR reduction across 32 lanes into lane 0. ----
    // After each stage lane 0 carries the XOR of the lanes folded in so far.
    // TODO(HW): the SFPSHFT2 mod-4 encoding here is a placeholder — the
    // immediate-form unroll pattern in horizontal_reduce_max
    // (ckernel_sfpu_reduce.h) is the production-quality reference. Validate
    // on device before relying on cross-run comparisons.
    for (int stage = 4; stage >= 0; --stage)
    {
        TTI_SFPMOV(0, LREG_H, LREG_TMP, 0);
        TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, 4);
        TTI_SFPXOR(0, LREG_TMP, LREG_H, 0);
    }

    // ---- Stash lane 0 of LREG_H in DEST[0][row 0] so MATH can read it. ----
    TT_SFPSTORE(LREG_H, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

    // ---- Drain SFPU + DEST writes before reading DEST from scalar code. ----
    // SFPSTORE goes through the SFPU pipeline; without this, dbg_get_array_row
    // can race the pending DEST write and return the previous tile's value.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

    // ---- Read DEST row 0 via debug array path (8 dwords). ----
    // Lane 0's value lives at rd_data[0] for INT32 DEST rows.
    std::uint32_t rd_data[8];
    dbg_get_array_row(dbg_array_id::DEST, /*row_addr=*/0, rd_data);
    const std::uint32_t h = rd_data[0];

    // ---- Publish to L1: hash first, then ready flag. ----
    // Two volatile writes to distinct addresses are not reorderable w.r.t.
    // each other, and the L1 cache is write-through on BH, so UNPACK's poll
    // (with invalidate_l1_cache()) observes hash before ready.
    *reinterpret_cast<volatile std::uint32_t*>(l1_hash_addr)  = h;
    *reinterpret_cast<volatile std::uint32_t*>(l1_ready_addr) = 1u;
}

} // namespace ckernel::sfpu
