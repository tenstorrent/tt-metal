// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// ============================================================================
// SFPU binary-with-broadcast kernel (BCAST_COL / BCAST_ROW)
// ============================================================================
//
// Tile layout in dest register (SFPU-addressable):
//   A 32x32 tile is 4 faces of 16x16 values, arranged:
//       Face 0 (tile rows  0-15 | tile cols  0-15)
//       Face 1 (tile rows  0-15 | tile cols 16-31)
//       Face 2 (tile rows 16-31 | tile cols  0-15)
//       Face 3 (tile rows 16-31 | tile cols 16-31)
//
//   One SFPLOAD/SFPSTORE moves 4 dest rows x 8 cols (32 lanes). Dest addresses
//   are in units of "4-row x 8-col" slots:
//       addr +0 -> rows 0-3,  cols  0-7 of the face
//       addr +2 -> rows 0-3,  cols  8-15
//       addr +4 -> rows 4-7,  cols  0-7
//       ...
//       addr +14 -> rows 12-15, cols 8-15
//       addr +16 -> start of next face
//
// SFPU lane numbering in a register (32 lanes = 4 sub-rows x 8 columns):
//       lane[i * 8 + c]  i in {0..3}, c in {0..7}
//   After SFPLOAD of rows r..r+3 cols c0..c0+7:
//       LReg[i * 8 + c] = data[r + i][c0 + c]
//
// Broadcast model:
//   - BCAST_COL: column 0 of the bcast tile is replicated across all 32 tile
//                columns, then eltwise-combined with the data tile.
//   - BCAST_ROW: row 0 of the bcast tile is replicated across all 32 tile
//                rows, then eltwise-combined with the data tile.
//
// Implementation strategy (BCAST_COL):
//   For each 4-row band of the source/dest tiles:
//     (1) Load col 0 (8 values, one per SFPU column) into LREG_DATA.
//         After load, only "SFPU column 0" in each 8-lane sub-vector carries
//         the useful value (the data broadcast target); the other 7 SFPU
//         columns hold d[row][cols 1..7] which we must discard.
//     (2) Zero out SFPU cols 1..7 in each 8-lane group by multiplying with a
//         precomputed lane mask LREG_MASK (1.0 in col 0, 0.0 in cols 1..7).
//     (3) Broadcast col-0 value to all 8 SFPU columns via 3 stages of
//         "rotate-by-k + add" using SFPSHFT2_MOD1_SUBVEC_SHFLROR1:
//             stage 1: ROR by 1, ADD  -> col {0,1} filled
//             stage 2: ROR by 2, ADD  -> col {0..3} filled
//             stage 3: ROR by 4, ADD  -> col {0..7} filled
//         This works because the masked-out lanes are 0.0, so ADD is
//         equivalent to OR.
//     (4) For each of the 4 target (col-group, face) slots in this row band,
//         load the data tile's corresponding slot, perform the elementwise
//         op against the broadcast value, and store the result.
//
// All "compute" sequences that are identical across row bands are recorded
// into the SFPU replay buffer so the loop body is short.
//
// NOTE: The current production usage only requires BCAST_COL. BCAST_ROW is
//       stubbed out behind a static_assert so a future commit can implement
//       it symmetrically without breaking the interface.

// ============================================================================
// Layout constants
// ============================================================================

enum class SfpuBcastDim : std::uint8_t
{
    BCAST_COL = 0,
    BCAST_ROW = 1,
};

// One dest-register tile occupies 64 addr units (4 faces x 16 addr/face).
constexpr std::uint32_t DEST_TILE_SIZE_RAW = 64;

// Per-face address offsets (in dest addr units):
//   Face 0 upper-left,  Face 1 upper-right,
//   Face 2 lower-left,  Face 3 lower-right.
constexpr std::uint32_t FACE0_BASE = 0;
constexpr std::uint32_t FACE1_BASE = 16;
constexpr std::uint32_t FACE2_BASE = 32;
constexpr std::uint32_t FACE3_BASE = 48;

// Within a face, each "row band" (4 dest rows) starts every 4 addr units.
constexpr std::uint32_t ROW_BAND_STRIDE             = 4;
constexpr std::uint32_t NUM_ROW_BANDS_PER_FACE_HALF = 4; // 16 rows / 4 rows-per-band
constexpr std::uint32_t ODD_COLS_OFFSET             = 2; // addr +2 selects cols 8..15 of a face

// ============================================================================
// LReg assignments used across the kernel
// ============================================================================
//
//   LREG_MASK  (LREG6)  - one-time lane mask: 1.0 in SFPU col 0, 0.0 elsewhere.
//                         Built once in _sfpu_binary_bcast_init_().
//   LREG_BCAST (LREG0)  - broadcast value (after load+mask+rotate-add).
//   LREG_DATA  (LREG1)  - data tile value loaded per-store-slot.
//   LREG_TMP   (LREG2)  - scratch for rotate-and-add.
//
// LREG3 / LREG4 / LREG5 are free for future use (e.g. interleaving two bands
// to hide latency).

constexpr std::uint32_t LREG_BCAST = p_sfpu::LREG0;
constexpr std::uint32_t LREG_DATA  = p_sfpu::LREG1;
constexpr std::uint32_t LREG_TMP   = p_sfpu::LREG2;
constexpr std::uint32_t LREG_MASK  = p_sfpu::LREG6;

// SFPSHFT2 Mod1 encoding: rotate right by 1 within each 8-lane sub-vector.
constexpr std::uint32_t SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 3;

// SFPCONFIG target index used with imm-mask to force LReg[11] = 1.0 on
// specific SFPU instances (= specific "SFPU columns" within each 8-lane
// group) and 0.0 on the others. Bit N of the mask corresponds to SFPU
// instance N; the low 8 bits control the 8 SFPU columns.
constexpr std::uint32_t SFPCONFIG_TARGET_LREG11      = 11;
constexpr std::uint32_t SFPCONFIG_MOD_SET_LREG11     = 8;
constexpr std::uint32_t SFPCONFIG_MASK_INSTANCE_COL0 = 0x0001;

// Replay-buffer slots.
constexpr std::uint32_t REPLAY_SLOT_BROADCAST = 0;  // col-0 -> all-cols broadcast
constexpr std::uint32_t REPLAY_LEN_BROADCAST  = 12; // see _record_broadcast_replay_

// ============================================================================
// Helpers
// ============================================================================

// Build LREG_MASK = {1.0, 0, 0, 0, 0, 0, 0, 0} repeated across all 4 sub-rows
// of the register. Strategy: write 1.0 into the mask register only on SFPU
// instance 0 (= SFPU col 0 of each 8-lane sub-vector) via SFPSETCC guarded
// by a purpose-built LReg[11] marker.
//
// SFPCONFIG(imm_mask, 11, 8) sets LReg[11] = (current value of LReg[0]) but
// only on SFPU instances selected by imm_mask; other instances keep whatever
// LReg[11] previously held. Bit N of imm_mask selects SFPU instance N.
// Because only selected instances are overwritten, we must issue two
// SFPCONFIG passes: first to zero LReg[11] everywhere, then to set it to a
// nonzero marker on the target instance.
//
// Notes on SFPLOADI Mod0:
//   SFPLOADI_MOD0_UPPER (8)   -> writes high 16 bits of each lane's u32
//   SFPLOADI_MOD0_LOWER (10)  -> writes low  16 bits of each lane's u32
// FP32 0x3F800000 = 1.0, so high=0x3F80, low=0x0000.
inline void _build_lane_mask_col0_()
{
    constexpr std::uint32_t ALL_INSTANCES_MASK = 0x5555; // every SFPU instance

    // Start with LREG_MASK = 0.0 on all 32 lanes.
    TTI_SFPMOV(0, p_sfpu::LCONST_0, LREG_MASK, 0);

    // Step 1: clear LReg[11] on all instances.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPCONFIG(ALL_INSTANCES_MASK, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);

    // Step 2: set LReg[11] = 1.0 on SFPU instance 0 only.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPCONFIG(SFPCONFIG_MASK_INSTANCE_COL0, SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11);

    // Under "LReg[11] != 0" (true only on SFPU col 0 of each 8-lane group),
    // write 1.0 into LREG_MASK. Elsewhere LREG_MASK keeps its 0.0 from above.
    TTI_SFPSETCC(0, p_sfpu::LREG11, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_UPPER, 0x3F80);
    TTI_SFPLOADI(LREG_MASK, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPENCC(0, 0, 0, 0);
}

// Broadcast SFPU col 0 of LREG_BCAST to all 8 SFPU columns within each
// 8-lane sub-vector. Assumes SFPU cols 1..7 of each group are already 0.0
// (i.e. caller has applied LREG_MASK). Executes a log2(8) = 3-stage
// "rotate-k + add" fan-out using SFPSHFT2_MOD1_SUBVEC_SHFLROR1.
//
// This exact sequence is recorded into the replay buffer by
// _record_broadcast_replay_() and emitted inline here for the code
// path that doesn't use replay.
inline void _broadcast_col0_within_groups_inline_()
{
    // --- Stage 1: rotate current LREG_BCAST by 1, add -> {0,1} filled
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;

    // --- Stage 2: rotate current LREG_BCAST by 2 (ROR1 twice), add -> {0..3} filled
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;

    // --- Stage 3: rotate current LREG_BCAST by 4 (ROR1 four times), add -> {0..7} filled
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
}

// Emit one elementwise op between LREG_DATA and LREG_BCAST, result in LREG_DATA.
// LREG_BCAST is preserved so the same broadcast value can be reused across all
// 4 col-group stores in a row band.
template <BinaryOp BINOP>
inline void _apply_binop_(std::uint32_t data_lreg, std::uint32_t bcast_lreg, std::uint32_t dest_lreg)
{
    if constexpr (BINOP == BinaryOp::ADD)
    {
        // dest = data + bcast
        TT_SFPADD(data_lreg, p_sfpu::LCONST_1, bcast_lreg, dest_lreg, 0);
        TTI_SFPNOP; // required after SFPADD on Wormhole
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        // dest = data - bcast   <=>   dest = data * 1.0 + (-bcast)
        // SFPMOV with Mod1 = SFPMOV_MOD1_NEGATE (= 1) copies with sign-bit flip,
        // which works for FP32 and sign-magnitude integers. Write to a scratch
        // so the broadcast value stays intact for the other 3 slots in the band.
        TTI_SFPMOV(0, bcast_lreg, LREG_TMP, 1 /* SFPMOV_MOD1_NEGATE */);
        TT_SFPADD(data_lreg, p_sfpu::LCONST_1, LREG_TMP, dest_lreg, 0);
        TTI_SFPNOP;
    }
    else if constexpr (BINOP == BinaryOp::MUL)
    {
        // dest = data * bcast + 0
        TT_SFPMUL(data_lreg, bcast_lreg, p_sfpu::LCONST_0, dest_lreg, 0);
        TTI_SFPNOP;
    }
    else
    {
        static_assert(BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }
}

// Record the mask + stage1 (ROR by 1) + stage2 (ROR by 2) portion of the
// broadcast sequence into the replay buffer. Stage 3 (ROR by 4) is emitted
// inline by _broadcast_stage3_inline_() because keeping replay slots short
// leaves room for other kernels' replay entries if/when combined.
//
// Sequence recorded (12 SFPU ops):
//    1  SFPMUL    LREG_BCAST *= LREG_MASK   (zero SFPU cols 1..7 in-place)
//    2  SFPNOP
//    3  SFPSHFT2  LREG_TMP = ROR1(LREG_BCAST)
//    4  SFPNOP
//    5  SFPADD    LREG_BCAST += LREG_TMP
//    6  SFPNOP
//    7  SFPSHFT2  LREG_TMP = ROR1(LREG_BCAST)
//    8  SFPNOP
//    9  SFPSHFT2  LREG_TMP = ROR1(LREG_TMP)
//   10  SFPNOP
//   11  SFPADD    LREG_BCAST += LREG_TMP
//   12  SFPNOP
inline void _record_broadcast_replay_()
{
    lltt::record(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);

    // (1) Mask: zero out SFPU cols 1..7.
    TTI_SFPMUL(LREG_BCAST, LREG_MASK, p_sfpu::LCONST_0, LREG_BCAST, 0);
    TTI_SFPNOP;

    // Stage 1: ROR by 1, add.
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;

    // Stage 2: ROR by 2 (ROR1 twice), add.
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
}

// Emit the "rotate by 4, add" tail stage inline (not in the replay buffer),
// because inserting it pushes the recorded length past what we want to
// guarantee fits in a single slot on all SKUs.
inline void _broadcast_stage3_inline_()
{
    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP;
}

// ============================================================================
// BCAST_COL: per row-band compute kernel
// ============================================================================
//
// For one 4-row band of the tile (identified by the source/dest addresses
// pointing at col 0 of that band):
//   1. Load col-0 values (SFPU col 0 of each 8-lane group carries the useful
//      data; cols 1..7 are noise from adjacent tile cols).
//   2. Mask + broadcast (replay + inline stage 3).
//   3. For each of the 4 target col-group slots in this band (face-left even
//      cols, face-left odd cols, face-right even cols, face-right odd cols),
//      load the data tile, apply the binop, store to the output tile.
template <BinaryOp BINOP>
inline void _process_col_bcast_row_band_(
    std::uint32_t bcast_col0_addr, std::uint32_t left_face_addr, std::uint32_t right_face_addr, std::uint32_t data_tile_offset, std::uint32_t out_tile_offset)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    // (1) Load col 0 values: 4 dest rows x 8 SFPU cols. Only SFPU col 0 in
    //     each 8-lane group holds the column-0 values we want; cols 1..7
    //     hold data at tile-cols 1..7 (we'll zero them out with the mask).
    TT_SFPLOAD(LREG_BCAST, IM, ADDR_MOD_3, bcast_col0_addr);

    // (2) Broadcast: run recorded mask+ROR1+ROR2 sequence, then ROR4 tail.
    lltt::replay(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);
    _broadcast_stage3_inline_();

    // (3) For each of the 4 target col-group slots in this row band, combine
    //     with the data tile and store. Slots are expressed as tile-local
    //     offsets (face-base + in-face row-band offset + even/odd col-group).
    const std::uint32_t slots[4] = {
        left_face_addr,                    // left face, cols 0-7
        left_face_addr + ODD_COLS_OFFSET,  // left face, cols 8-15
        right_face_addr,                   // right face, cols 0-7
        right_face_addr + ODD_COLS_OFFSET, // right face, cols 8-15
    };

    for (std::uint32_t s = 0; s < 4; s++)
    {
        // Load data tile's corresponding slot.
        TT_SFPLOAD(LREG_DATA, IM, ADDR_MOD_3, data_tile_offset + slots[s]);

        // Apply binop: LREG_DATA op LREG_BCAST -> LREG_DATA (LREG_BCAST preserved).
        _apply_binop_<BINOP>(LREG_DATA, LREG_BCAST, LREG_DATA);

        // Store result.
        TT_SFPSTORE(LREG_DATA, IM, ADDR_MOD_3, out_tile_offset + slots[s]);
    }
}

// Top-level BCAST_COL driver for a single 32x32 tile.
template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_col_full_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    const std::uint32_t data_base  = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = dst_index_out * DEST_TILE_SIZE_RAW;

    // Upper tile half (tile rows 0-15): face 0 = left, face 1 = right.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;

        _process_col_bcast_row_band_<BINOP>(
            /* bcast_col0_addr  */ bcast_base + FACE0_BASE + band_off,
            /* left_face_addr   */ FACE0_BASE + band_off,
            /* right_face_addr  */ FACE1_BASE + band_off,
            /* data_tile_offset */ data_base,
            /* out_tile_offset  */ out_base);
    }

    // Lower tile half (tile rows 16-31): face 2 = left, face 3 = right.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        const std::uint32_t band_off = band * ROW_BAND_STRIDE;

        _process_col_bcast_row_band_<BINOP>(
            /* bcast_col0_addr  */ bcast_base + FACE2_BASE + band_off,
            /* left_face_addr   */ FACE2_BASE + band_off,
            /* right_face_addr  */ FACE3_BASE + band_off,
            /* data_tile_offset */ data_base,
            /* out_tile_offset  */ out_base);
    }
}

// ============================================================================
// Public API
// ============================================================================

template <bool APPROXIMATION_MODE, BinaryOp BINOP, SfpuBcastDim BCAST_DIM>
inline void _calculate_sfpu_binary_bcast_full_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    (void)APPROXIMATION_MODE; // no approximation path differs for ADD/SUB/MUL

    if constexpr (BCAST_DIM == SfpuBcastDim::BCAST_COL)
    {
        _calculate_sfpu_binary_bcast_col_full_tile_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
    }
    else
    {
        static_assert(BCAST_DIM == SfpuBcastDim::BCAST_COL, "BCAST_ROW is not yet implemented in the SFPU binary-bcast kernel");
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, SfpuBcastDim BCAST_DIM>
inline void _sfpu_binary_bcast_init_()
{
    (void)APPROXIMATION_MODE;
    (void)BINOP;

    // Zero-stride address modifiers for straight SFPLOAD / SFPSTORE use.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (BCAST_DIM == SfpuBcastDim::BCAST_COL)
    {
        // Build persistent col-0 lane mask in LREG_MASK.
        _build_lane_mask_col0_();

        // Record the broadcast helper (mask + ROR1 + ROR2) into replay slot 0.
        _record_broadcast_replay_();
    }
}

} // namespace sfpu
} // namespace ckernel
