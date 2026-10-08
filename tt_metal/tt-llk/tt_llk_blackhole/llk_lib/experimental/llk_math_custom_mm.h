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
inline void custom_mm_configure_addrmod(const bool m1_group)
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
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4); // Last MVMUL if finalization enabled, resets SrcA and SrcB,
                          // moves Dest to the beginning of the just multiplied tile

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 32, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5); // Move SrcB to 32 in preparation for finalization ELWADD

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6); // Move to next face in all three regs, used for final accumulation

    if (m1_group)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 1, .clr = 0, .cr = 0},
            .dest = {.incr = 16, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_5); // M 1 finalize: next SrcB row, from a tile's face 0 to its face 1

        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 1, .clr = 0, .cr = 0},
            .dest = {.incr = ADDR_MOD_2_DEST_INCR - 16, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_6); // M 1 finalize: next SrcB row, from a tile's face 1 to the next tile
    }

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7); // NOP, has to be ADDR_MOD_7 to match with address mods configured in silu
                          // for fusing matmul and activation
}

// At M 1 with more than one output tile, the finalize moves every tile's two partial rows to consecutive SrcB rows and
// adds each back with a row-broadcast ELWADD, so the MOVD2B hold comes once per call (replay: 4 tiles of moves, 2 of adds)
inline bool custom_mm_m1_group(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    return operandB_face_r_dim == 1 && ct_dim > 1;
}

constexpr std::uint32_t custom_mm_m1_moves      = 3;
constexpr std::uint32_t custom_mm_m1_adds       = custom_mm_m1_moves + 8;
constexpr std::uint32_t custom_mm_m1_replay_len = custom_mm_m1_adds + 4;

inline void custom_mm_record_m1_finalize()
{
    for (std::uint32_t t = 0; t < 4; t++)
    {
        TTI_MOVD2B(0, 0, ADDR_MOD_5, p_movd2a::MOV_1_ROW, 8);
        TTI_MOVD2B(0, 0, ADDR_MOD_6, p_movd2a::MOV_1_ROW, 8);
    }
    for (std::uint32_t t = 0; t < 2; t++)
    {
        TTI_ELWADD(0, 1, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_5, 0);
        TTI_ELWADD(0, 1, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_6, 0);
    }
}

template <bool split_acc>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    const bool m1_group                = custom_mm_m1_group(operandB_face_r_dim, ct_dim);
    const std::uint32_t replay_buf_len = m1_group ? custom_mm_m1_replay_len : operandB_face_r_dim == 8 ? 11 : 9;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        [operandB_face_r_dim, m1_group]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

            if (m1_group)
            {
                custom_mm_record_m1_finalize();
                return;
            }

            // Finalization phase
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // F1 @ F3 => 16 (24 if split_acc) (clear none,
                                                             // as both SrcA and SrcB banks are required for finalization)

            // We will exploit AddDst for finalization to calculate:
            // Dst = partial (in dest) + (split partial (moved to SrcB) + 0 (zeroed SrcA))
            // This way we avoid additional set of moves and move only one set of partials to SrcB
            // We have to move them to SrcB as we have to zero out the other Src register
            // and that can only be SrcA as it is not reused
            // Since we are going to keep reusing SrcB,
            // we have to move partials in lower half of SrcB as upper half contains inputs that are being reused

            // Move the 4 rows of the F0 split partial to rows 32+ of SrcB for finalization
            TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8);
            // Move the 4 rows of the F1 split partial to rows 48+ of SrcB for finalization
            TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8);

            // Move lower 4 rows if they exist
            if (operandB_face_r_dim == 8)
            {
                TTI_MOVD2B(0, 0 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4);
                TTI_MOVD2B(0, 16 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8 + 4);
            }

            // Zero SrcA as it is not reused and we need to add zero to split partials in SrcB
            TTI_ZEROSRC(0, 1, 0, 1);

            // F0 partial in dest + F0 split partial in SrcB (row 32) => final F0 in dest
            TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
            // F1 partial in dest + F1 split partial in SrcB (row 48) => final F1 in dest (clear only SrcA as B is reused)
            TTI_ELWADD(1, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
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
    custom_mm_configure_addrmod<transpose, split_acc, dense_packing>(custom_mm_m1_group(operandB_face_r_dim, ct_dim));
    custom_mm_configure_mop<split_acc>(operandB_face_r_dim, ct_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void custom_mm_finalize_m1_(const std::uint32_t ct_dim)
{
    constexpr std::uint32_t moves = ckernel::math::replay_buf_offset + custom_mm_m1_moves;
    constexpr std::uint32_t adds  = ckernel::math::replay_buf_offset + custom_mm_m1_adds;

    // The last tile's MVMULs keep SrcA and SrcB; that SrcA, zeroed, is the other operand of every ELWADD
    lltt::replay(ckernel::math::replay_buf_offset, 3);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
    TTI_ZEROSRC(0, 1, 0, 1);
    for (std::uint32_t first = 0; first < ct_dim; first += 4)
    {
        lltt::replay(moves, ct_dim - first < 4 ? 2 * (ct_dim - first) : 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);
    std::uint32_t t = 0;
    for (; t + 2 < ct_dim; t += 2)
    {
        lltt::replay(adds, 4);
    }
    lltt::replay(adds, t + 2 == ct_dim ? 3 : 1);
    TTI_ELWADD(3, 1, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_3, 0);
}

template <bool finalize = true>
inline void _llk_math_custom_mm_(
    const std::uint32_t operandB_face_r_dim, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim = 1)
{
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Run mop kt_dim - 1 times, the last inner dim is handled separately to choose whether to run finalization or not
    for (std::uint32_t i = 0; i < kt_dim - 1; i++)
    {
        TTI_MOP(1, 0, 0);
    }

    if constexpr (finalize)
    {
        if (custom_mm_m1_group(operandB_face_r_dim, ct_dim))
        {
            for (std::uint32_t i = 0; i < ct_dim - 1; i++)
            {
                lltt::replay(ckernel::math::replay_buf_offset, 3);
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
            }
            custom_mm_finalize_m1_(ct_dim);
            return;
        }
        // Run the full replay ct_dim - 1 times as it has the full finalization sequence for cases where SrcB is reused
        for (std::uint32_t i = 0; i < ct_dim - 1; i++)
        {
            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
        }
        // Run the full replay - the last ELWADD as it differs on the last ct_dim iteration and is issued directly after
        lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len - 1);
        TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
    }
    else
    {
        // Just run another mop iteration if not finalizing
        TTI_MOP(1, 0, 0);
    }
}
