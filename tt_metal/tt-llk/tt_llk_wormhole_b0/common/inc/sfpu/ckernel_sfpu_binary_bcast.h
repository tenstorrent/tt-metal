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
#include "sfpu/ckernel_sfpu_load_config.h"

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
// Implementation strategy (BCAST_ROW):
//   Two variants are provided, selectable at compile time via
//   _SFPU_BINARY_BCAST_ROW_USE_TRANSPOSE (default = 0, i.e. brute-force).
//
//   Variant A (brute-force, DEFAULT):
//     Phase I  - Build a "broadcast scratch" in the result tile: replicate
//                bcast row 0 across all 32 rows of the result tile. Uses
//                SFPTRANSP to extract a single tile row into an LREG with
//                that row's 8 values replicated across all 4 sub-rows, then
//                stores that LREG to all 8 row bands of the target col-group.
//     Phase II - Straightforward per-slot eltwise: for each of the 32 slots
//                (8 row bands x 4 col-groups), SFPLOAD data, SFPLOAD result
//                (which holds the replicated bcast row), apply binop, store.
//     Pros: compute loop is a trivial in-place eltwise (no per-slot bcast
//           dance). Cons: Phase I issues 32 extra SFPSTOREs to prime the
//           scratch, and Phase II doubles the loads vs. a pure in-place path.
//
//   Variant B (SFPTRANSP + in-place, when _USE_TRANSPOSE == 1):
//     For each 4-row band of data, load data's 4 col-groups into LREG0..3
//     and bcast's row-0 band's 4 col-groups into LREG4..7. SFPTRANSP. Now
//     LREG_k holds data row (base+k), and LREG_{4+k} holds bcast row k
//     (only LREG4 - bcast row 0 - is useful). Op LREG_k with LREG4 for each
//     k in [0..3]. SFPTRANSP back, store.
//     Pros: no scratch priming, fewer SFPLOAD/SFPSTOREs overall.
//     Cons: per-band SFPTRANSP overhead; more intricate register plumbing.
//
//   NOTE: Both variants currently target Wormhole B0 only. The SFPTRANSP
//         semantics and LREG aliasing have been validated by analogy with
//         ckernel_sfpu_reshuffle_rows.h but both variants should be
//         sanity-checked on real silicon before being enabled for Blackhole.

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
// BCAST_COL path:
//   LREG_MASK  (LREG6)  - one-time lane mask: 1.0 in SFPU col 0, 0.0 elsewhere.
//                         Built once in _sfpu_binary_bcast_init_().
//   LREG_BCAST (LREG0)  - broadcast value (after load+mask+rotate-add).
//   LREG_TMP   (LREG2)  - scratch for rotate-and-add inside the broadcast
//                         sequence; reused as -bcast scratch in the SUB path.
//   LREG_DATA{0..3} (LREG1, LREG3, LREG4, LREG5) - 4 data tile values
//                         pre-loaded per row band so the per-slot binops can
//                         pipeline back-to-back without inter-instruction
//                         NOPs (each binop writes a distinct LREG).
//
// BCAST_ROW path:
//   Uses LREG0..LREG7 transiently during SFPTRANSP phases. LREG6 is NOT
//   preserved as a persistent mask (SFPTRANSP clobbers LREG4..LREG7 window);
//   BCAST_ROW does not need the col-0 lane mask.

constexpr std::uint32_t LREG_BCAST = p_sfpu::LREG0;
constexpr std::uint32_t LREG_TMP   = p_sfpu::LREG2;
constexpr std::uint32_t LREG_MASK  = p_sfpu::LREG6;

// Four distinct LREGs for pipelined per-slot data values in BCAST_COL.
// They must be mutually distinct AND distinct from LREG_BCAST/LREG_TMP/LREG_MASK
// so that the 4 binops can be issued back-to-back without inter-op NOPs.
constexpr std::uint32_t LREG_DATA0 = p_sfpu::LREG1;
constexpr std::uint32_t LREG_DATA1 = p_sfpu::LREG3;
constexpr std::uint32_t LREG_DATA2 = p_sfpu::LREG4;
constexpr std::uint32_t LREG_DATA3 = p_sfpu::LREG5;

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

// BCAST_ROW: choose between "brute-force scratch prime" (default, 0) and
// "SFPTRANSP + in-place" (1). Override via the build system if needed.
#ifndef _SFPU_BINARY_BCAST_ROW_USE_TRANSPOSE
#define _SFPU_BINARY_BCAST_ROW_USE_TRANSPOSE 0
#endif

// BCAST_ROW col-group offsets within a face pair (face N + face N+1), in
// dest addr units. A "col group" is one 4-row x 8-col SFPLOAD/SFPSTORE slot.
// For the upper tile half, faces are 0 (left, cols 0-15) and 1 (right, cols
// 16-31). For each face, +0 selects cols 0-7 and +2 selects cols 8-15.
constexpr std::uint32_t COL_GROUP_OFFSETS[4] = {
    0,                            // face N   cols 0-7
    ODD_COLS_OFFSET,              // face N   cols 8-15
    FACE1_BASE,                   // face N+1 cols 0-7
    FACE1_BASE + ODD_COLS_OFFSET, // face N+1 cols 8-15
};

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

// Record the mask + stage1 (ROR by 1) + stage2 (ROR by 2) portion of the
// broadcast sequence into the replay buffer. Stage 3 (ROR by 4) is emitted
// inline by _broadcast_stage3_with_data_prefetch_() because keeping replay
// slots short leaves room for other kernels' replay entries if/when
// combined, and because stage 3 interleaves per-slot SFPLOADs into its
// SFPSHFT2 latency slots (data addresses are per-call so can't be recorded).
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
//
// The 4 SFPSHFT2 ops form a strict serial chain through LREG_TMP (each is
// 2-cycle and writes the LREG the next op reads), so the 4 inner latency
// slots must be filled. They are filled by interleaving the 4 independent
// per-slot data loads (LREG_DATA0..3 are disjoint from LREG_BCAST/LREG_TMP)
// - this lets us fold the data prefetch into stage 3 for free instead of
// issuing 4 SFPLOADs after the broadcast finishes.
//
// Postcondition: on return, LREG_BCAST holds the fully-broadcast value and
// LREG_DATA0..3 hold the 4 per-slot data values, both ready to be read by
// the next instruction (the trailing SFPNOP drains the final SFPADD's
// 2-cycle latency on LREG_BCAST so callers don't need to insert their own).
inline void _broadcast_stage3_with_data_prefetch_(std::uint32_t data_addr0, std::uint32_t data_addr1, std::uint32_t data_addr2, std::uint32_t data_addr3)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    TTI_SFPSHFT2(0, LREG_BCAST, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA0, IM, ADDR_MOD_3, data_addr0); // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA1, IM, ADDR_MOD_3, data_addr1); // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA2, IM, ADDR_MOD_3, data_addr2); // fills LREG_TMP latency
    TTI_SFPSHFT2(0, LREG_TMP, LREG_TMP, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TT_SFPLOAD(LREG_DATA3, IM, ADDR_MOD_3, data_addr3); // fills LREG_TMP latency
    TTI_SFPADD(LREG_BCAST, p_sfpu::LCONST_1, LREG_TMP, LREG_BCAST, 0);
    TTI_SFPNOP; // drain SFPADD's 2-cycle latency on LREG_BCAST so callers can read it next cycle
}

// ============================================================================
// BCAST_COL: per row-band compute kernel
// ============================================================================
//
// For one 4-row band of the tile (identified by the source/dest addresses
// pointing at col 0 of that band):
//   1. Load col-0 values (SFPU col 0 of each 8-lane group carries the useful
//      data; cols 1..7 are noise from adjacent tile cols).
//   2. Mask + broadcast stages 1+2 from the replay buffer, then stage 3
//      (ROR4 + ADD) inline with the 4 per-slot data SFPLOADs interleaved
//      into stage 3's RAW-latency slots (saves 4 inner NOPs per row band).
//   3. Issue the 4 binops back-to-back; each writes a distinct LREG and
//      reads LREG_BCAST, so they pipeline without inter-op NOPs.
//   4. Stream out 4 SFPSTOREs; the first is 4 instructions after the SFPADD
//      that wrote LREG_DATA0, so no leading NOP is needed.
template <BinaryOp BINOP>
inline void _process_col_bcast_row_band_(
    std::uint32_t bcast_col0_addr, std::uint32_t left_face_addr, std::uint32_t right_face_addr, std::uint32_t data_tile_offset, std::uint32_t out_tile_offset)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    // Slot addresses within this row band (tile-local: face-base + in-face
    // row-band offset + even/odd col-group).
    const std::uint32_t slot0 = left_face_addr;                    // left face, cols 0-7
    const std::uint32_t slot1 = left_face_addr + ODD_COLS_OFFSET;  // left face, cols 8-15
    const std::uint32_t slot2 = right_face_addr;                   // right face, cols 0-7
    const std::uint32_t slot3 = right_face_addr + ODD_COLS_OFFSET; // right face, cols 8-15

    // (1) Load col 0 values: 4 dest rows x 8 SFPU cols. Only SFPU col 0 in
    //     each 8-lane group holds the column-0 values we want; cols 1..7
    //     hold data at tile-cols 1..7 (we'll zero them out with the mask).
    TT_SFPLOAD(LREG_BCAST, IM, ADDR_MOD_3, bcast_col0_addr);

    // (2) Broadcast: replay records mask+ROR1+ROR2 (stage 1+2). Stage 3
    //     (ROR4+ADD) is emitted inline with the 4 data SFPLOADs interleaved
    //     into the SFPSHFT2 chain's latency slots, so LREG_DATA0..3 are
    //     ready by the time the per-slot binops below start.
    lltt::replay(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);
    _broadcast_stage3_with_data_prefetch_(data_tile_offset + slot0, data_tile_offset + slot1, data_tile_offset + slot2, data_tile_offset + slot3);

    // (3) Pipelined per-slot binop. The 4 ops are pairwise independent
    //     (distinct dest LREGs, all reading LREG_BCAST and a distinct
    //     LREG_DATAk), so they issue back-to-back with no NOPs between
    //     them. The helper above already drained the stage-3 SFPADD's
    //     2-cycle latency on LREG_BCAST, and the subsequent 4 SFPSTOREs
    //     provide the trailing 2-cycle drain implicitly - no NOPs needed.
    if constexpr (BINOP == BinaryOp::ADD)
    {
        TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA0, 0);
        TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA1, 0);
        TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA2, 0);
        TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_BCAST, LREG_DATA3, 0);
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        // data - bcast = data + (-bcast). Negate bcast once into LREG_TMP
        // (free after the broadcast sequence) then reuse for all 4 adds.
        TTI_SFPMOV(0, LREG_BCAST, LREG_TMP, 1 /* SFPMOV_MOD1_NEGATE */);
        TTI_SFPADD(LREG_DATA0, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA0, 0);
        TTI_SFPADD(LREG_DATA1, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA1, 0);
        TTI_SFPADD(LREG_DATA2, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA2, 0);
        TTI_SFPADD(LREG_DATA3, p_sfpu::LCONST_1, LREG_TMP, LREG_DATA3, 0);
    }
    else if constexpr (BINOP == BinaryOp::MUL)
    {
        TTI_SFPMUL(LREG_DATA0, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA0, 0);
        TTI_SFPMUL(LREG_DATA1, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA1, 0);
        TTI_SFPMUL(LREG_DATA2, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA2, 0);
        TTI_SFPMUL(LREG_DATA3, LREG_BCAST, p_sfpu::LCONST_0, LREG_DATA3, 0);
    }
    else
    {
        static_assert(BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }

    // (5) Streamed stores. The first SFPSTORE issues 4 cycles after the
    //     SFPADD/SFPMUL that wrote LREG_DATA0 - well past the 2-cycle
    //     latency window - so no leading NOP is required.
    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_3, out_tile_offset + slot0);
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_3, out_tile_offset + slot1);
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_3, out_tile_offset + slot2);
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_3, out_tile_offset + slot3);
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
// BCAST_ROW: row-0 splat primitive
// ============================================================================
//
// Produces an LREG whose 4 sub-rows all equal bcast row 0 for a single
// 8-column col-group. The resulting LREG can then be SFPSTOREd to any
// 4-row-band slot (at the matching col-group address) to fill those 4 rows
// with copies of bcast row 0.
//
// Method:
//   1. SFPLOAD bcast rows 0..3 of this col-group into LREG_SCRATCH_A.
//      Only sub-row 0 is useful (= bcast[0][c0..c0+7]); sub-rows 1..3 hold
//      bcast[1..3][c0..c0+7] (noise).
//   2. Build a helper LREG_BCAST_SRC where ALL 4 sub-rows contain the
//      useful row-0 value. Done via the same SFPTRANSP trick used in
//      ckernel_sfpu_reshuffle_rows.h: load the same 4-row band into 4
//      different LREGs (LREG0..3) all pointing at the SAME col-group
//      address. After SFPTRANSP, LREG_k contains "tile row k of the
//      col-group", and critically its 8 column values end up replicated
//      across all 4 sub-rows of LREG_k (because each of the 4 source
//      loads was identical, so sub-row k of each source was identical,
//      so the transposed output is a constant across sub-rows).
//
// HARDWARE NOTE: step (2)'s replication-across-sub-rows claim depends on
// SFPTRANSP's exact semantics when all 4 inputs are identical 4-row-band
// loads. This needs silicon validation. If it fails, fallback is to use
// sub-row-predicated SFPMOVs (SFPSETCC on LReg[10] or equivalent).
//
// The helper takes the address of the col-group to extract (relative to
// the bcast tile base) and leaves row 0 of that col-group replicated
// across all 4 sub-rows of LREG0. (LREG1..3 hold rows 1..3 replicated -
// unused noise; LREG4..7 are also clobbered by SFPTRANSP and must not
// be live across this call.)
inline void _bcast_row_load_row0_replicated_to_lreg0_(std::uint32_t bcast_col_group_addr)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    // Load the same 4-row band (rows 0..3 of the col-group) into LREG0..3.
    // Pre-transpose each LREG_k has: LREG_k[sub_r][c] = bcast[sub_r][c0+c].
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, bcast_col_group_addr);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, bcast_col_group_addr);
    TT_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_3, bcast_col_group_addr);
    TT_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, bcast_col_group_addr);

    // Transpose: post-SFPTRANSP LREG_k[sub_r] = pre-LREG_{sub_r}[k]. Since
    // all pre-LREGs are identical, pre-LREG_{sub_r}[k] = bcast[k][c0..c0+7]
    // for any sub_r. So post-LREG_k[sub_r] = bcast[k][c0..c0+7] for every
    // sub_r - i.e. row k replicated across all 4 sub-rows of LREG_k.
    //
    // LREG0 is the one we want (= bcast row 0 replicated). Caller stores
    // straight from LREG0 - no SFPMOV needed.
    TTI_SFPTRANSP(0, 0, 0, 0);
}

// ============================================================================
// BCAST_ROW: Variant A (brute-force scratch prime) driver
// ============================================================================

// PRECONDITION: dst_index_out MUST be distinct from dst_index_data.
//   Phase I writes broadcasted row-0 scratch values into the output tile
//   BEFORE Phase II reads the data tile. If out == data, Phase I will
//   overwrite the input data and produce incorrect results. Callers that
//   need an in-place (out == data) semantics must either route through a
//   scratch tile or use the transpose variant
//   (_calculate_sfpu_binary_bcast_row_full_tile_transpose_), which is
//   in-place-safe.
template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_row_full_tile_bruteforce_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    const std::uint32_t data_base  = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = dst_index_out * DEST_TILE_SIZE_RAW;

    // bcast row 0 always lives in face 0 (upper-left) and face 1 (upper-right).
    // For each of the 4 col-groups (cg0..cg3 covering cols 0-31), extract
    // row 0 replicated across sub-rows, then SFPSTORE to all 8 row bands
    // (4 upper + 4 lower) at the matching col-group.
    //
    // Result tile after Phase I: every 4-row-band at col-group c contains 4
    // copies of bcast[0][c0(c)..c0(c)+7].

    // --- Phase I: prime the output tile with bcast row 0 replicated -------
    for (std::uint32_t cg = 0; cg < 4; cg++)
    {
        const std::uint32_t cg_off = COL_GROUP_OFFSETS[cg];

        // LREG0: bcast row 0 of this col-group, replicated across all sub-rows.
        _bcast_row_load_row0_replicated_to_lreg0_(bcast_base + FACE0_BASE + cg_off);

        // Store LREG0 into all 4 upper row bands (face 0 or face 1 - same cg_off).
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
        {
            const std::uint32_t addr = out_base + FACE0_BASE + band * ROW_BAND_STRIDE + cg_off;
            TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, addr);
        }
        // Store LREG0 into all 4 lower row bands (face 2 or face 3 - same cg_off
        // relative to face2 base, since face2 is the "face0-equivalent" of lower
        // half).
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
        {
            const std::uint32_t addr = out_base + FACE2_BASE + band * ROW_BAND_STRIDE + cg_off;
            TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, addr);
        }
    }

    // --- Phase II: in-place eltwise: out = data OP out ----------------------
    // For each of the 8 row bands, process all 4 col-groups in a pipelined
    // batch:
    //   - LREG0..3 hold the 4 data values for this band's col-groups
    //   - LREG4..7 hold the 4 bcast values (= primed row 0 contents) for the
    //     same col-groups
    //   - 4 binops issue back-to-back (each writes a distinct LREG0..3, all
    //     read distinct LREG4..7), no inter-op NOPs needed
    //   - 4 SFPSTOREs drain the write-back without a leading NOP
    //
    // For SUB the bcast registers are negated in-place first - the 4 SFPMOVs
    // are also pairwise independent and pipeline cleanly.
    constexpr std::uint32_t FACE_HALF_BASES[2] = {FACE0_BASE, FACE2_BASE};
    for (std::uint32_t half = 0; half < 2; half++)
    {
        const std::uint32_t half_base = FACE_HALF_BASES[half];
        for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
        {
            const std::uint32_t band_off = band * ROW_BAND_STRIDE;
            const std::uint32_t slot0    = half_base + band_off + COL_GROUP_OFFSETS[0];
            const std::uint32_t slot1    = half_base + band_off + COL_GROUP_OFFSETS[1];
            const std::uint32_t slot2    = half_base + band_off + COL_GROUP_OFFSETS[2];
            const std::uint32_t slot3    = half_base + band_off + COL_GROUP_OFFSETS[3];

            // Pre-load 4 data values into LREG0..3.
            TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, data_base + slot0);
            TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, data_base + slot1);
            TT_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_3, data_base + slot2);
            TT_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, data_base + slot3);

            // Pre-load 4 bcast values (= primed scratch in the out tile) into LREG4..7.
            TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_3, out_base + slot0);
            TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_3, out_base + slot1);
            TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_3, out_base + slot2);
            TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_3, out_base + slot3);

            if constexpr (BINOP == BinaryOp::ADD)
            {
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
                TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
                TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
            }
            else if constexpr (BINOP == BinaryOp::SUB)
            {
                // In-place negate each bcast LREG. The 4 SFPMOVs write
                // distinct LREGs and pipeline cleanly. The first dependent
                // SFPADD reads LREG4 several cycles after its SFPMOV.
                TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG4, 1 /* NEGATE */);
                TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG5, 1);
                TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG6, 1);
                TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG7, 1);
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
                TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
                TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
            }
            else if constexpr (BINOP == BinaryOp::MUL)
            {
                TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
                TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
                TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
            }
            else
            {
                static_assert(
                    BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
            }

            TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, out_base + slot0);
            TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_3, out_base + slot1);
            TT_SFPSTORE(p_sfpu::LREG2, IM, ADDR_MOD_3, out_base + slot2);
            TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_3, out_base + slot3);
        }
    }
}

// ============================================================================
// BCAST_ROW: Variant B (SFPTRANSP + in-place) driver
// ============================================================================
//
// For each 4-row band of the data tile (8 bands total across both halves):
//   1. Load data's 4 col-groups into LREG0..3.
//   2. Load bcast row-0 band's 4 col-groups into LREG4..7.
//   3. SFPTRANSP (transposes {LREG0..3} and {LREG4..7} independently).
//      Now:
//        LREG_k (k in 0..3) = data row (band_base + k), 32 cols across sub-rows.
//        LREG_{4+k}         = bcast row k, 32 cols across sub-rows.
//        We only care about LREG4 (bcast row 0); LREG5..7 are noise.
//   4. For each k in 0..3: LREG_k = LREG_k OP LREG4  (data row op bcast row 0).
//   5. SFPTRANSP back.
//   6. Store LREG0..3 to the 4 col-groups of the output band.

template <BinaryOp BINOP>
inline void _process_row_bcast_data_band_transpose_(
    std::uint32_t data_tile_base, std::uint32_t bcast_tile_base, std::uint32_t out_tile_base, std::uint32_t face_pair_base, std::uint32_t band_off)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::FP32;

    const std::uint32_t data_slot_base  = data_tile_base + face_pair_base + band_off;
    const std::uint32_t bcast_row0_base = bcast_tile_base + FACE0_BASE; // bcast row 0 always in upper-left
    const std::uint32_t out_slot_base   = out_tile_base + face_pair_base + band_off;

    // (1) Load data's 4 col-groups into LREG0..3.
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, data_slot_base + COL_GROUP_OFFSETS[3]);

    // (2) Load bcast row-0 band's 4 col-groups into LREG4..7.
    TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_3, bcast_row0_base + COL_GROUP_OFFSETS[3]);

    // (3) Transpose: LREG0..3 become the 4 data rows; LREG4..7 become the 4
    //     bcast rows (of which only LREG4 = bcast row 0 is useful).
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (4) In-register eltwise: LREG_k = LREG_k OP LREG4 for k in 0..3.
    //
    // Pipelining: the 4 binops all read LREG4 and a distinct LREG_k, and each
    // writes a distinct LREG_k. They are pairwise independent, so we can
    // issue them back-to-back with no NOPs between. Only a single trailing
    // NOP is needed before SFPTRANSP reads LREG0..3 (2-cycle SFPADD/SFPMUL
    // latency). LREG5..7 are post-transpose junk and free for use as scratch.
    if constexpr (BINOP == BinaryOp::ADD)
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        // data - bcast = data + (-bcast). Negate bcast once into LREG5
        // (post-transpose junk register), then reuse for all 4 adds.
        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 1 /* SFPMOV_MOD1_NEGATE */);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::MUL)
    {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    }
    else
    {
        static_assert(BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }
    TTI_SFPNOP; // single drain before SFPTRANSP reads LREG0..3.

    // (5) Transpose back (LREG0..3 return to their original 4-row-band layout,
    //     now containing the per-row eltwise results).
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (6) Store LREG0..3 to the 4 col-groups of the output band.
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[1]);
    TT_SFPSTORE(p_sfpu::LREG2, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[2]);
    TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_3, out_slot_base + COL_GROUP_OFFSETS[3]);
}

template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_row_full_tile_transpose_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    const std::uint32_t data_base  = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = dst_index_out * DEST_TILE_SIZE_RAW;

    // Upper half (face 0/1): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _process_row_bcast_data_band_transpose_<BINOP>(data_base, bcast_base, out_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }

    // Lower half (face 2/3): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _process_row_bcast_data_band_transpose_<BINOP>(data_base, bcast_base, out_base, FACE2_BASE, band * ROW_BAND_STRIDE);
    }
}

// ============================================================================
// BCAST_ROW: variant dispatch
// ============================================================================

template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_row_full_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
#if _SFPU_BINARY_BCAST_ROW_USE_TRANSPOSE
    _calculate_sfpu_binary_bcast_row_full_tile_transpose_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
#else
    _calculate_sfpu_binary_bcast_row_full_tile_bruteforce_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
#endif
}

// ============================================================================
// Public API
// ============================================================================

template <BinaryOp BINOP, SfpuBcastDim BCAST_DIM>
inline void _calculate_sfpu_binary_bcast_full_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    if constexpr (BCAST_DIM == SfpuBcastDim::BCAST_COL)
    {
        _calculate_sfpu_binary_bcast_col_full_tile_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
    }
    else
    {
        _calculate_sfpu_binary_bcast_row_full_tile_<BINOP>(dst_index_data, dst_index_bcast, dst_index_out);
    }
}

template <SfpuBcastDim BCAST_DIM>
inline void _sfpu_binary_bcast_init_()
{
    // Initialize SFPU config register (matches sibling SFPU init helpers,
    // e.g. _init_add_top_row_, init_reduce_*). Required before any replay
    // recording or SFPCONFIG-based lane-mask setup below.
    _init_sfpu_config_reg();

    // Zero-stride address modifier for straight SFPLOAD / SFPSTORE use.
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
    // BCAST_ROW: no persistent state needed - both variants operate purely
    // on per-tile LREG contents via SFPTRANSP + SFPLOAD/SFPSTORE.
}

} // namespace sfpu
} // namespace ckernel
