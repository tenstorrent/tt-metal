// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

/*************************************************************************
 * LLK MATH FAST TILIZE (Tilize single input using both unpackers and packer)
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim and num_units must match the ones given to the unpacker (all unit_dim usage notes from the unpacker also apply here)
 * dst_index is the index of the tile inside the destination register to write to
 * both dest modes are supported (although 32 bit mode is supported by intentionally ignoring it for both math and pack unless src regs are TF32)
 * only DstSync::SyncHalf is supported
 * tiles are split across halves of the active dest bank (effectively quarters since DstSync::SyncHalf)
 * so nothing except fast tilize should be using that dest bank
 *************************************************************************/

/**
 * @brief Program the address-mod slots for fast tilize, including the jumps to and from the bottom-face dest offset.
 *
 * ADDR_MOD_1/ADDR_MOD_2 are the standard 4-row (MOVB2D) and 8-row (MOVA2D) steps; ADDR_MOD_3/ADDR_MOD_0 hold the
 * forward/backward dest jumps used to interleave top and bottom faces, computed from the per-bank bottom-face offset
 * (which halves for TF32) and the unit_dim.
 *
 * @param unpack_dst_format: Destination data format (DataFormat enum underlying value); TF32 halves the bottom-face offset.
 * @param unit_dim: Number of tiles processed per iteration (selects the jump increments); must match the unpacker.
 */
inline void _llk_math_fast_tilize_addrmod_config_(const std::uint32_t unpack_dst_format, const std::uint32_t unit_dim)
{
    // standard addrmod that follows MOVB2D
    addr_mod_t {
        .srcb = {.incr = 4},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_1);

    // standard addrmod that follows MOVA2D
    addr_mod_t {
        .srca = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

    // next two addrmods are mostly used for jumping to and from the offset for the bottom faces
    // offset for the bottom faces is always half the number of rows in the dest bank (512 / 2 for 16bit and 256 / 2 for 32bit since DstSync is always Half)
    std::uint32_t bottom_face_offset = (unpack_dst_format == to_underlying(DataFormat::Tf32) ? 256 : 512) / 2;
    // unit_dim 1 copies 2 faces before jumping so at the moment of the jump dest RWC is
    // 2*16 (two faces) -  8 (number of rows moved by current instruction)
    // unit_dim 2 copies 4 faces before jumping so at the moment of the jump dest RWC is
    // 4*16 (four faces) - 8 (number of rows moved by current instruction)
    std::uint8_t unit_dim_1_forward_jump = bottom_face_offset - (1 * (TILE_NUM_FACES / 2) * FACE_R_DIM - 8);
    std::uint8_t unit_dim_2_forward_jump = bottom_face_offset - (2 * (TILE_NUM_FACES / 2) * FACE_R_DIM - 8);

    // jumping back to the offset for the next tile is logically -bottom_face_offset if dest RWC is at the correct offset for the bottom faces of the next tile
    // only catch is the need to compensate for the current instruction, for unit_dim 1 that is MOVA2D while for unit_dim 2 and 3 that is MOVB2D
    std::int16_t unit_dim_1_backward_jump = -bottom_face_offset + 8;
    std::int16_t unit_dim_2_backward_jump = -bottom_face_offset + 4;

    if (unit_dim == 1)
    {
        // this follows MOVA2D in src and jumps to the offset for the bottom faces (for unit_dim 1 and 2, for unit_dim 3 that is handled the other way)
        addr_mod_t {
            .srca = {.incr = 8},
            .dest = {.incr = unit_dim_1_forward_jump},
        }
            .set(ADDR_MOD_3);

        // this jumps back to the offset for the next tile, RWCs for source registers are reset separately when clearing dvalids
        addr_mod_t {
            .dest = {.incr = unit_dim_1_backward_jump},
        }
            .set(ADDR_MOD_0);
    }
    else
    {
        // this follows MOVA2D in src and jumps to the offset for the bottom faces (for unit_dim 1 and 2, for unit_dim 3 that is handled the other way)
        addr_mod_t {
            .srca = {.incr = 8},
            .dest = {.incr = unit_dim_2_forward_jump},
        }
            .set(ADDR_MOD_3);

        // this jumps back to the offset for the next tile, RWCs for source registers are reset separately when clearing dvalids
        addr_mod_t {
            .dest = {.incr = unit_dim_2_backward_jump},
        }
            .set(ADDR_MOD_0);
    }
}

/**
 * @brief Program the fast-tilize MOP with the MOVA2D (8-row) and MOVB2D (4-row) move templates used by the block loop.
 */
inline void _llk_math_fast_tilize_mop_config_()
{
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,
        false,
        TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0),
        TT_OP_NOP,
        TT_OP_NOP);

    tmp.program();
}

/**
 * @brief Initialize the math thread for fast tilize: programs address mods and the move MOP.
 *
 * For non-TF32 formats, switches to CFG state 1 and clears the FP32 dest-accumulation bit so MOVA2D/MOVB2D fully
 * ignore FP32 dest mode (a HW quirk). Only DstSync::SyncHalf is supported.
 *
 * @param unpack_dst_format: Destination data format (DataFormat enum underlying value); TF32 keeps FP32 dest mode.
 * @param unit_dim: Number of tiles processed per iteration; must match the unpacker.
 * @note On the unpack thread, pair with @ref _llk_unpack_fast_tilize_init_ which feeds the top/bottom faces into SrcA/SrcB.
 * @note On the pack thread, pair with @ref _llk_pack_fast_tilize_init_ (same unit_dim) which drains the split dest halves.
 * @note Call @ref _llk_math_fast_tilize_uninit_ to restore the FP32 dest-mode/state changes; run with @ref _llk_math_fast_tilize_block_.
 */
inline void _llk_math_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t unit_dim)
{
    // even though MOVA2D and MOVB2D are supposed to ignore ALU_ACC_CTRL_Fp32_enabled some parts still rely on it (not sure why)
    // it would be easier if they just fully respected ALU_ACC_CTRL_Fp32_enabled but it's a hardware quirk
    // so in non Tf32 cases, clear it to fully ignore FP32 dest mode
    // not sure why it doesn't work if CFG_STATE_ID_StateID is not 1
    if (unpack_dst_format != to_underlying(DataFormat::Tf32))
    {
        TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, 1);
        TTI_NOP;
        TTI_NOP;
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }

    // everything else is quite standard math init
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_fast_tilize_addrmod_config_(unpack_dst_format, unit_dim);

    _llk_math_fast_tilize_mop_config_();
}

/**
 * @brief Uninitialize after fast tilize, restoring the FP32 dest-accumulation mode and CFG state that init changed.
 *
 * @tparam is_fp32_dest_acc_en: FP32 dest-accumulation mode to restore (must match the surrounding context).
 * @param unpack_dst_format: Destination data format (DataFormat enum underlying value); only non-TF32 needs restoring.
 * @note Reverses @ref _llk_math_fast_tilize_init_.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_tilize_uninit_(const std::uint32_t unpack_dst_format)
{
    // if ALU_ACC_CTRL_Fp32_enabled was previously cleared, restore it
    // still not sure why this CFG_STATE_ID_StateID manipulation is needed
    if (unpack_dst_format != to_underlying(DataFormat::Tf32))
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
        TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, 0);
        TTI_NOP;
        TTI_NOP;
    }
}

/**
 * @brief Tilize a block of tiles into the destination register, moving top and bottom faces into split dest halves.
 *
 * For each unit, copies the top faces (via MOVA2D from SrcA) and the bottom faces (via MOVB2D from SrcB) into the
 * two halves of the active dest bank, using the configured address mods to jump between offsets. The unit_dim and
 * num_faces select which move/clear sequence runs (unit_dim 3 uses an explicit dest offset for the forward jump).
 * Only DstSync::SyncHalf is supported, and nothing else should use the active dest bank.
 *
 * @param dst_index: Tile index into the destination register to write to.
 * @param unpack_dst_format: Destination data format (DataFormat enum underlying value); affects the bottom-face offset.
 * @param unit_dim: Number of tiles processed per iteration; must match the unpacker.
 * @param num_units: Number of units processed in this call.
 * @param num_faces: Number of faces per tile (must be 2 or 4).
 * @note Call @ref _llk_math_fast_tilize_init_ with matching unpack_dst_format and unit_dim before this function.
 * @note On the unpack thread, @ref _llk_unpack_fast_tilize_block_ must feed the tiles into SrcA/SrcB.
 * @note On the pack thread, @ref _llk_pack_fast_tilize_block_ drains the split dest halves into tilized L1 output.
 */
inline void _llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units,
    const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4");
    // split dest and write the top faces in the first half and the bottom faces in the second half (or more precisely quarter, since dest sync half)
    // make life easier by lying to set_dst_write_addr that tile shape is 32x16 so correct stride is obtained for dst_index
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // srcA has the full tile, copy the top faces first
            // inside mop:
            // for (uint j = 0; j < 3; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 3 - 1, 0x0);
            // finish with the top faces and jump to the offset for the bottom faces
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
            // copy the bottom faces
            // inside mop:
            // for (uint j = 0; j < 3; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 3 - 1, 0x0);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 0);
            // clear just srcA dvalid since it's the only one set by the unpacker for unit_dim 1 and src RWCs
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else if (unit_dim == 2 && num_faces == 4)
        {
            // srcA has the top faces (4 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 7; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 7 - 1, 0x0);
            // finish with the top faces and jump to the offset for the bottom faces
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
            // srcB has the bottom faces (4 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 15 - 1, 0xFFFF);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
            // clear both dvalids and src RWCs
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else if (unit_dim == 2 && num_faces == 2)
        {
            // srcA has the top 8 rows
            // inside mop:
            // for (uint j = 0; j < 4; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 4 - 1, 0x0);
            // srcB has the bottom 8 rows
            // inside mop:
            // for (uint j = 0; j < 8; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 8 - 1, 0xFFFF);
            // done with this set of two tiles, clear dvalids and src RWCs
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else if (unit_dim == 3)
        {
            // srcA has the top 8 rows of the top faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 6; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 6 - 1, 0x0);
            // srcB has the bottom 8 rows of the top faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 12; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 12 - 1, 0xFFFF);
            // done with the top faces, clear dvalids and src RWCs, next banks contain bottom faces
            // also clear dest RWC since we use dest offset for forward jump here
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
            // don't have enough address mods to have unit_dim 3 forward jump so dest offset is used here
            std::uint32_t top_face_offset = dst_index + i * 3; // copy 3 tiles per iteration
            // offset to the bottom is the number of tiles that fit into the dest bank
            // since half size faces are specified, this gets into the correct position in the second half
            std::uint32_t bottom_face_offset = top_face_offset + (unpack_dst_format == to_underlying(DataFormat::Tf32) ? 4 : 8);
            math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(bottom_face_offset);
            // srcA has the top 8 rows of the bottom faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 6; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 6 - 1, 0x0);
            // srcB has the bottom 8 rows of the bottom faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 11; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 11 - 1, 0xFFFF);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
            // clear both dvalids and src RWCs
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
    }
    math::clear_dst_reg_addr();
}
