// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

template <BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod_custom()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 8},
        // The increment field is 6 bits wide, so 0x3F & -8 encodes a step of
        // -8 and effectively rewinds SrcB back by one face.
        .srcb = {.incr = 0x3F & -8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 24},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param num_faces: Number of faces to process (1, 2, or 4)
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type>
inline void _llk_math_eltwise_binary_init_custom_(const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod_custom<src_b_bcast_type>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

// This helper is intentionally specialized even though the init surface is
// generic. The current in-tree caller is blocked sub+bcast(col), so the
// instruction sequence is hard-wired to ELWSUB with SrcB column broadcast and
// SrcB reuse across ct_dim.
//
// Each 16x16 face pair (one face-row) is processed by four ELWSUBs (two per
// face; each ELWSUB covers an aligned 8x16 block). A full 32x32 tile has two
// face-rows (F0/F1 and F2/F3); a 16x32 tiny tile has a single face-row (F0/F1).
// @param ct_dim       Number of SrcA tiles that reuse the single broadcast SrcB tile.
// @param tensor_shape Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
// @param dst_index    Absolute dest tile slot where this block-row's ct_dim tiles begin. Multi-tile-row
//                     callers (e.g. the fuser LoopBlockRow driver) advance this per block-row so each row
//                     lands on its own dest slots; single-tile-row callers leave it at 0.
inline void _llk_math_sub_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Two faces make up one face-row; a full tile has two of them, a tiny tile one.
    const std::uint32_t num_face_rows = tensor_shape.num_faces_r_dim;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB);

    for (std::uint32_t tile = 0; tile < ct_dim; tile++)
    {
        // Position the dest write pointer at the start of this tile's 32x32 dest
        // slot (stride 64 rows). The packer indexes tiles at the full 32x32 tile
        // stride regardless of num_faces, so tiles must be written one-per-slot.
        // The ADDR_MOD dest increments below only advance 32 rows for a 16x32 tiny
        // tile (num_faces=2) and 64 rows for a full 32x32 tile (num_faces=4); the
        // per-slot base + counter reset makes both land on the right slot.
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + tile);
        math::clear_dst_reg_addr();

        for (std::uint32_t face_row = 0; face_row < num_face_rows; face_row++)
        {
            // Even face (F0 / F2) - F0
            TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_0, 0);
            // Even face - next, rewind SrcB by one face to the start of the broadcast tile.
            TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_1, 0);

            // Odd face (F1 / F3) - F0
            TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_0, 0);
            // Odd face - advance SrcB to the next face-row (or next tile on the last face-row).
            TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_2, 0);
        }

        // Move to the next SrcA tile while keeping the broadcasted SrcB tile
        // available for reuse on the next outer iteration.
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
    }

    // Release the reused SrcB tile once the whole ct_dim block is consumed.
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB);

    // Restore the dest base offset so the next op starts from tile slot 0.
    math::clear_dst_reg_addr();
}
