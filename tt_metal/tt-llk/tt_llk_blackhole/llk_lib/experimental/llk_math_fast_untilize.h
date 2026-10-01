// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Math.
//
// Consumes 2/3/4 fast_untilize unpack tiles (four real SrcA dvalids per tile,
// plus zero SrcB dvalids only for native fp32 DEST) and writes a Dst layout
// matching llk_pack_fast_untilize.h:
//
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Each Dst row is one 16-datum face row. The packer then uses
// ALL_INTF_ACTIVE + DST_ACCESS_STRIDED_MODE to read rows R, R+16, R+32, R+48
// and emit a contiguous 64-datum chunk.
//
// Math is the layout bridge: unpack delivers faces one at a time through SrcA,
// while pack wants four interface rows arranged so one full-width PACR can emit
// a contiguous row-major strip. These copies do not reduce input payload reads;
// they make the pack-side access pattern wider and regular.

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "experimental/llk_fast_untilize_common.h"
#include "llk_math_common.h"

namespace ckernel
{

inline void _llk_math_fast_untilize_configure_addrmod_()
{
    // Explicit MOVA2D source/destination row immediates drive all placement.
    // Keep RWCs unchanged so each face copy starts from SrcA rows 0/8 and
    // writes to the immediate Dst row selected below.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_4);

    // Native fp32 DEST uses ELWADD as SrcA + zero-SrcB -> DEST. ELWADD has no
    // SrcA row immediate, so the first copy advances SrcA to row 8 and the
    // second copy consumes that row. DEST placement still uses immediates.
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_5);
}

template <bool configure_remap = true>
inline void _llk_math_fast_untilize_init_()
{
    if constexpr (configure_remap)
    {
        // Same Dst read remap required by pack_untilize/fast_tilize paths so packer
        // STRIDED_MODE sees a 16-row stride through Dst.
        _llk_math_reconfig_remap_(true);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_fast_untilize_configure_addrmod_();
}

inline void _llk_math_fast_untilize_copy_face_(const std::uint32_t dst_row)
{
    // MOVA2D moves eight rows at a time, so one 16-row face is placed with two
    // copies. The source row immediates pick SrcA rows 0 and 8; ADDR_MOD_4 keeps
    // SrcA/DEST RWCs fixed so the destination row immediate controls placement.
    TTI_MOVA2D(0, 0, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, dst_row);
    TTI_MOVA2D(0, 8, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, dst_row + 8);

    // SrcA has been consumed for this face. Clear its dvalid so unpack can
    // reuse the bank for the next face without waiting on stale validity state.
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
}

inline void _llk_math_fast_untilize_copy_face_fp32_(const std::uint32_t dst_row)
{
    // Native fp32 DEST cannot use MOVA2D here; use ELWADD with zero SrcB as a
    // copy. ADDR_MOD_5 advances SrcA between the two half-face operations.
    TTI_ELWADD(0, p_elwise::DEST_ACCUM_DIS, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, dst_row);
    TTI_ELWADD(0, p_elwise::DEST_ACCUM_DIS, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, dst_row + 8);

    // ELWADD consumes both source operands, including the dummy zero SrcB face.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_untilize_copy_tile_(const std::uint32_t tile_index)
{
    const std::uint32_t top_row    = FAST_UNTILIZE_TOP_STRIP_ROW_OFFSET + tile_index * FAST_UNTILIZE_TILE_STRIDE_ROWS;
    const std::uint32_t bottom_row = FAST_UNTILIZE_BOTTOM_STRIP_ROW_OFFSET + tile_index * FAST_UNTILIZE_TILE_STRIDE_ROWS;

    // The unpacker presents each tile as F2, F3, F0, F1. Copy bottom faces
    // first, then top faces, so SrcA is consumed in arrival order while DEST is
    // arranged into the shared fast-untilize two-strip layout.
    if constexpr (is_fp32_dest_acc_en)
    {
        _llk_math_fast_untilize_copy_face_fp32_(bottom_row);
        _llk_math_fast_untilize_copy_face_fp32_(bottom_row + FAST_UNTILIZE_PHASE_ROWS);
        _llk_math_fast_untilize_copy_face_fp32_(top_row);
        _llk_math_fast_untilize_copy_face_fp32_(top_row + FAST_UNTILIZE_PHASE_ROWS);
    }
    else
    {
        _llk_math_fast_untilize_copy_face_(bottom_row);
        _llk_math_fast_untilize_copy_face_(bottom_row + FAST_UNTILIZE_PHASE_ROWS);
        _llk_math_fast_untilize_copy_face_(top_row);
        _llk_math_fast_untilize_copy_face_(top_row + FAST_UNTILIZE_PHASE_ROWS);
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_untilize_block_(const std::uint32_t dst_index, const std::uint32_t block_ct_dim)
{
    LLK_ASSERT(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast-untilize supports block_ct_dim 2, 3, or 4");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);

    _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(0);
    _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(1);
    if (block_ct_dim >= 3)
    {
        _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(2);
    }
    if (block_ct_dim >= 4)
    {
        _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(3);
    }

    math::clear_dst_reg_addr();
}

inline void _llk_math_fast_untilize_uninit_()
{
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}

} // namespace ckernel
