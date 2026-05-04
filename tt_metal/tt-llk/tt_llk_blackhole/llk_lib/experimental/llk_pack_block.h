// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

/*************************************************************************
 * LLK PACK BLOCK CONTIGUOUS
 *
 * Packs multiple tiny tiles from sparse DEST (Tile32x32 slot convention)
 * to dense L1 (contiguous output) in a single _llk_pack_ call.
 *
 * MOP structure:
 *   OUTER = num_tiles (runtime-patchable via mop_cfg[0])
 *   INNER = 1
 *   START_OP  = REPLAY(all-but-last PACRs for one tile)
 *   last_inner= last PACR with ADDR_MOD_2 (non-last tile)
 *   last_outer= last PACR with ADDR_MOD_1 + Last=1 (final tile)
 *   END_OP0   = INCADCZW(W++) to advance to next Tile32x32 slot
 *   END_OP1   = SETADCZW(Z=0) to reset Z for next tile
 *
 * REPLAY buffer stores num_faces * pacrs_per_face - 1 PACR instructions
 * encoding the complete per-tile face traversal. The final PACR of each
 * tile lives in the MOP template (last_inner / last_outer) so it can
 * carry the Last=1 bit on the final tile.
 *
 * Precondition: _llk_pack_init_ or _llk_pack_configure_addrmod_ +
 * set_packer_strides must have been called to establish the normal
 * pack ADDR_MOD_0/1/2 and strides. This function only replaces the MOP.
 *************************************************************************/

// Program the REPLAY buffer and MOP for block-contiguous packing.
template <bool zero_output = false>
inline void _llk_pack_block_contiguous_mop_config_(
    [[maybe_unused]] const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    const std::uint32_t PACK_INTF_SEL = face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

    const std::uint32_t pacrs_per_face = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
    const std::uint32_t total_pacrs    = num_faces * pacrs_per_face;

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(total_pacrs >= 1, "At least 1 PACR required per tile");
    LLK_ASSERT(total_pacrs <= 16, "Replay buffer overflow: max 16 PACRs per tile");

    // Guard the subtraction to avoid unsigned underflow — the compiler's
    // intrinsic range checker on __builtin_rvtt_ttreplay rejects UINT32_MAX
    // even in dead code paths.
    const std::uint32_t replay_len = (total_pacrs > 1) ? (total_pacrs - 1) : 0;

    // --- Load REPLAY buffer with the per-tile PACR sequence ---
    // Contains: for each face, (pacrs_per_face - 1) PACRs with ADDR_MOD_0
    // followed by 1 PACR with ADDR_MOD_2 (face transition: Y clear, Z += 1).
    // The very last PACR of the tile is NOT in the replay — it goes in the
    // MOP template as last_inner / last_outer.
    //
    // We use instrn_buffer writes (runtime values) instead of TTI (which
    // requires compile-time immediates) so that face_r_dim / num_faces can
    // be runtime parameters.
    if (replay_len > 0)
    {
        // Pre-compute the two PACR instruction words we'll need
        const std::uint32_t pacr_mod0 = TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            0);

        const std::uint32_t pacr_mod2 = TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            0);

        // Enter recording mode: capture next replay_len instructions
        lltt::record<lltt::NoExec>(0, replay_len);

        // Emit instructions into replay buffer via instrn_buffer (runtime-safe)
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            const bool is_last_face          = (face == num_faces - 1);
            const std::uint32_t replay_pacrs = is_last_face ? (pacrs_per_face - 1) : pacrs_per_face;

            for (std::uint32_t p = 0; p < replay_pacrs; p++)
            {
                const bool is_last_pacr_of_face = (p == pacrs_per_face - 1);
                ckernel::instrn_buffer[0]       = is_last_pacr_of_face ? pacr_mod2 : pacr_mod0;
            }
        }
    }

    // --- Program MOP ---
    // OUTER = num_tiles, INNER = 1
    // START_OP: replay the per-tile PACR sequence (all but last PACR)
    // last_inner: the final PACR of a non-last tile (ADDR_MOD_2, Last=0)
    // last_outer: the final PACR of the last tile (ADDR_MOD_1, Last=1)
    // END_OP0: W++ (next Tile32x32 slot)
    // END_OP1: Z=0 (reset for next tile's face traversal)

    const std::uint32_t start_op = (replay_len > 0) ? lltt::replay_insn(0, replay_len) : TT_OP_NOP;

    ckernel::ckernel_template tmp(
        1,        // OUTER (placeholder — overwritten by mop_cfg[0] in _llk_pack_block_contiguous_)
        1,        // INNER
        TT_OP_NOP // loop_op0 (unused: INNER=1 means only last_inner fires)
    );

    // last_inner: fires on every outer iteration except the last
    tmp.set_last_inner_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        ZERO_OUTPUT_FLAG,
        PACK_INTF_SEL,
        0,
        0,
        0,
        0,
        0)); // Last=0

    // last_outer: fires only on the very last outer iteration
    tmp.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        ZERO_OUTPUT_FLAG,
        PACK_INTF_SEL,
        0,
        0,
        0,
        0,
        1)); // Last=1

    // START_OP: replay per-tile PACRs (or NOP if tile has only 1 PACR)
    tmp.set_start_op(start_op);

    // END_OP0: advance W to next Tile32x32 DEST slot
    // END_OP1: reset Z for next tile's face traversal
    tmp.set_end_ops(
        TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),          // ch0_w += 1
        TT_OP_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001)); // ch0_z = 0

    tmp.program();
}

// Execute the block-contiguous pack.
// tile_index: starting tile in DEST (sets W counter)
// address: L1 destination address for the contiguous output block
// num_tiles: number of tiles to pack (1-8, runtime parameter)
//
// The outer loop count is patched via mop_cfg[0] before each MOP run,
// matching the pattern used by _llk_pack_set_mop_outer_loop_().
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_contiguous_(const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t num_tiles)
{
    set_dst_write_addr(tile_index);

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001); // Z = 0

    program_packer_destination(address);

    // Patch outer loop count to num_tiles before running the MOP.
    volatile std::uint32_t* mop_cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[0] = num_tiles;
    TTI_MOP(1, 0, 0);

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset Z/W
}
