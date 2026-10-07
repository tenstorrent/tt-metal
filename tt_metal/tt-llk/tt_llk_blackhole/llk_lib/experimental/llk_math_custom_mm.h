// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: any integer from 1 to 16
// kt_dim: any integer from 1 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

template <bool transpose, bool split_acc, bool dense_packing>
inline void custom_mm_configure_addrmod()
{
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR  = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR  = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = split_acc ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR  = dense_packing ? 32 : 64;

    addr_mod_t {
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Move to the next face in width dimension, if transpose that is face 2 instead of face 1

    addr_mod_t {
        .srca = {.incr = ADDR_MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_1_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1); // Move to the next face in inner dimension, if transpose that is face 1 instead of face 2,
                          // if split_acc next inner dim is stored at rows 8 and 24 instead of 0 and 16

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2); // Move to the next tile in width dimension,
                          // if dense_packing next tile starts at row 32 instead of 64

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3); // Move to the next inner dimension, resets everything

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5); // Finalize: from a tile's face 0 sum to its face 1 sum

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR - 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6); // Finalize: from a tile's face 1 sum to the next tile's face 0 sum

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7); // NOP, has to be ADDR_MOD_7 to match with address mods configured in silu
                          // for fusing matmul and activation
}

// The finalize adds each tile's split partials (rows 8 and 24, the second K half) to rows 0 and 16, for as many tiles at a
// time as the replay buffer holds: MOVD2B to SrcB rows 16 j and 16 j + 8, then one ELWADD pair per tile.
inline std::uint32_t custom_mm_finalize_group(const std::uint32_t operandB_face_r_dim)
{
    return operandB_face_r_dim == 8 ? 2 : 4;
}

inline std::uint32_t custom_mm_finalize_moves(const std::uint32_t operandB_face_r_dim)
{
    return operandB_face_r_dim == 8 ? 4 : 2;
}

template <bool split_acc, bool dense_packing>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    constexpr std::uint32_t tile_rows  = dense_packing ? 32 : 64;
    const std::uint32_t group          = custom_mm_finalize_group(operandB_face_r_dim);
    const std::uint32_t replay_buf_len = 3 + group * custom_mm_finalize_moves(operandB_face_r_dim) + 2;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        [operandB_face_r_dim, group]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

            for (std::uint32_t j = 0; j < group; j++)
            {
                TT_MOVD2B(0, 16 * j, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, j * tile_rows + 8);
                if (operandB_face_r_dim == 8)
                {
                    TT_MOVD2B(0, 16 * j + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, j * tile_rows + 12);
                }
                TT_MOVD2B(0, 16 * j + 8, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, j * tile_rows + 24);
                if (operandB_face_r_dim == 8)
                {
                    TT_MOVD2B(0, 16 * j + 12, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, j * tile_rows + 28);
                }
            }

            // Dst += 0 (SrcA, zeroed) + the moved partial, face 0 then face 1
            TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, 0);
            TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
        });

    // Top 3 MVMUL instructions from the replay buffer
    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, 3);
    // F1 @ F3 => 16 (24 if split_acc) (clear only SrcA as SrcB is reused)
    const std::uint32_t mvmul_end_tile = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
    // F1 @ F3 => 16 (24 if split_acc) (clear both SrcA and SrcB as this is the end if the block)
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);

    // Mop runs a single inner dim, perform ct_dim tile multiplications while reusing a single SrcB bank
    // For all ct_dim iterations except the last we run 4 MVMULs from mvmul_base and mvmul_end_tile
    // (clearing only SrcA and moving to next output tile in Dest)
    // For the last ct_dim iteration we run 4 MVMULs from mvmul_base and mvmul_end_block
    // (clearing both SrcA and SrcB and moving back to the beginning of Dest for next inner dim)

    ckernel_template tmp = ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
inline void _llk_math_custom_mm_init_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1)
{
    custom_mm_configure_addrmod<transpose, split_acc, dense_packing>();
    custom_mm_configure_mop<split_acc, dense_packing>(operandB_face_r_dim, ct_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void custom_mm_finalize_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    const std::uint32_t group = custom_mm_finalize_group(operandB_face_r_dim);
    const std::uint32_t moves = custom_mm_finalize_moves(operandB_face_r_dim);
    const std::uint32_t adds  = ckernel::math::replay_buf_offset + 3 + group * moves;

    // The last weight tile's SrcA, still held, is the zero operand of every ELWADD
    TTI_ZEROSRC(0, 1, 0, 1);
    for (std::uint32_t first = 0; first < ct_dim; first += group)
    {
        const std::uint32_t tiles = ct_dim - first < group ? ct_dim - first : group;
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_B);
        lltt::replay(ckernel::math::replay_buf_offset + 3, tiles * moves);
        for (std::uint32_t t = 0; t < tiles; t++)
        {
            if (first + t == ct_dim - 1)
            {
                lltt::replay(adds, 1);
                TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
            }
            else
            {
                lltt::replay(adds, 2);
            }
        }
    }
}

template <bool finalize = true>
inline void _llk_math_custom_mm_(
    const std::uint32_t operandB_face_r_dim, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim = 1)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Run mop kt_dim - 1 times, the last inner dim is handled separately to choose whether to run finalization or not
    for (std::uint32_t i = 0; i < kt_dim - 1; i++)
    {
        TTI_MOP(1, 0, 0);
    }

    if constexpr (finalize)
    {
        // The last inner dim as the MOP runs it, except that the last tile keeps SrcA and SrcB for the finalize
        for (std::uint32_t i = 0; i < ct_dim - 1; i++)
        {
            lltt::replay(ckernel::math::replay_buf_offset, 3);
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
        }
        lltt::replay(ckernel::math::replay_buf_offset, 3);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
        custom_mm_finalize_(operandB_face_r_dim, ct_dim);
    }
    else
    {
        // Just run another mop iteration if not finalizing
        TTI_MOP(1, 0, 0);
    }
}
