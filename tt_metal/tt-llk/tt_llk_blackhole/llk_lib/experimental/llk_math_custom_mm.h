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
#include "experimental/llk_custom_mm_common.h"
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::math;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32], or [32, 32] (a full in0 tile: in1 [32, 32], no transpose, the Default layout,
//                 no finalize; [32, 32] output tiles)
// in1 tile shape: [32, 32] or [32, 16] ([16, 32] with transpose), giving [{1, 2, 4, 8}, 32] or [{1, 2, 4, 8}, 16] output tiles
// rt_dim: 1, or any integer of 1 or more with a full in0 tile: output tile (r, c) at dst_index + r * ct_dim + c
// ct_dim: any integer from 1 to 34, as long as the ct_dim output tiles fit in the DEST sync region
//         (custom_mm_tile_rows of the CustomMmDestLayout each)
// kt_dim: any integer of 1 or more
// fidelity: LoFi only
// throttle: not supported

/**
 * @brief Whether ct_dim output tiles from dst_index fit in the DEST sync region.
 *
 * @tparam Dst: DEST sync mode the kernel runs with, values = <SyncHalf/SyncFull>
 * @tparam layout: DEST layout of the output tiles; dst_index counts tiles of its custom_mm_tile_rows.
 * @param dst_index: First DEST tile index, as passed to @ref _llk_math_custom_mm_.
 * @param ct_dim: Number of output column tiles.
 * @note For LLK asserts: reads the DEST accumulation mode from config.
 */
template <DstSync Dst, CustomMmDestLayout layout>
inline bool custom_mm_fits_dest(const std::uint32_t dst_index, const std::uint32_t ct_dim)
{
    constexpr std::uint32_t unit_rows = 16; // DstTileShape::Tile16x16
    return (dst_index + ct_dim) * custom_mm_tile_rows(layout) <= get_dest_max_tiles_rt<Dst, DstTileShape::Tile16x16>() * unit_rows;
}

template <bool transpose, CustomMmDestLayout layout>
inline void custom_mm_configure_addrmod()
{
    constexpr bool split_acc = custom_mm_split_acc(layout);
    constexpr bool compact   = layout == CustomMmDestLayout::Compact;

    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR  = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_0_DEST_INCR  = compact ? 8 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR  = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = (split_acc || compact) ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR  = custom_mm_tile_rows(layout);

    addr_mod_t {
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_0_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Move to the next face in width dimension, if transpose that is face 2 instead of face 1,
                          // if Compact output face 1 starts at row 8 instead of 16

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
        .set(ADDR_MOD_2); // Move to the next tile in width dimension, custom_mm_tile_rows down

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

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7); // NOP, has to be ADDR_MOD_7 to match with address mods configured in silu
                          // for fusing matmul and activation
}

// Narrow in1: each MVMUL steps to the next k-face in SrcA and SrcB, accumulating onto the same output rows (+8 if
// split_acc). Run after custom_mm_configure_addrmod; overrides its ADDR_MOD_0 and leaves ADDR_MOD_1 and 6 unused.
template <CustomMmDestLayout layout>
inline void custom_mm_configure_narrow_addrmod()
{
    constexpr std::uint8_t ADDR_MOD_0_DEST_INCR = custom_mm_split_acc(layout) ? 8 : 0;

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_0_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Move to the next face in inner dimension, F1 of in1 with or without transpose
}

inline void custom_mm_configure_narrow_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    load_replay_buf(
        ckernel::math::replay_buf_offset,
        custom_mm_replay_buf_len(operandB_face_r_dim, CUSTOM_MM_NARROW_IN1_NUM_FACES),
        [operandB_face_r_dim]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_0, 0 /*dst*/); // F0 @ F0 => 0

            // Finalization phase, as in custom_mm_configure_mop but with a single output face
            TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_4, 0 /*dst*/); // F1 @ F1 => 0 (8 if split_acc)

            // Move the 4 rows of the split partial to rows 32+ of SrcB for finalization
            TTI_MOVD2B(0 /*dest_32b_lo*/, 32 /*src*/, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8 /*dst*/);

            // Move lower 4 rows if they exist (SrcB is now at row 32)
            if (operandB_face_r_dim == 8)
            {
                TTI_MOVD2B(0 /*dest_32b_lo*/, 0 + 4 /*src*/, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4 /*dst*/);
            }

            // Zero SrcA as it is not reused and we need to add zero to the split partial in SrcB
            TTI_ZEROSRC(0 /*zero_val*/, 1 /*write_mode*/, 0 /*bank_mask*/, 1 /*src_mask*/);

            // Partial in dest + split partial in SrcB (row 32) => final output face in dest (clear only SrcA as B is reused)
            TTI_ELWADD(1 /*clear_dvalid*/, 1 /*dest_accum_en*/, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0 /*dst*/);
        });

    // The single top MVMUL from the replay buffer, then the end of tile or block MVMULs as in custom_mm_configure_mop
    const std::uint32_t mvmul_base      = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, 1 /*length*/);
    const std::uint32_t mvmul_end_tile  = TT_OP_MVMUL(p_setrwc::CLR_A, 0 /*instr_mod19*/, ADDR_MOD_2, 0 /*dst*/);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0 /*instr_mod19*/, ADDR_MOD_3, 0 /*dst*/);

    ckernel_template tmp = ckernel_template(1 /*outer_loop_len*/, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
}

// The fifteen MVMULs of a [32, 32] x [32, 32] tile pair the replay holds; the MOP issues the sixteenth (end of tile
// or end of k-row, as for a short in0 tile)
constexpr std::uint32_t CUSTOM_MM_FULL_IN0_REPLAY_LEN = 15;

// Address mods a [32, 32] in0 tile adds (over custom_mm_configure_addrmod's ADDR_MOD_0, 1, 4 and 5; 2 and 3 end the
// tile and the k-row as for a short in0 tile). An MVMUL multiplies 8 SrcB rows by one SrcA face into 8 DEST rows, so
// a tile pair is 16 of them. With in0 faces (row, k) at SrcB row 16 * (2 * row + k), in1 faces (k, col) at SrcA row
// 16 * (2 * k + col) and output faces (row, col) at DEST row 16 * (2 * row + col), the walk goes: the two 8-row
// halves of an in0 face, then the output face column, then the face row, then the k face:
//
//   k  row col half | SrcA SrcB DEST | then
//   0   0   0   0   |   0    0    0  | next half
//   0   0   0   1   |   0    8    8  | next column
//   0   0   1   0   |  16    0   16  | next half
//   0   0   1   1   |  16    8   24  | next face row
//   0   1   0   0   |   0   32   32  | next half
//   0   1   0   1   |   0   40   40  | next column
//   0   1   1   0   |  16   32   48  | next half
//   0   1   1   1   |  16   40   56  | next k face
//   1   ...           32/48 16-56 0-56, the same eight steps one k face on in both sources
//
// SrcA / SrcB counters wrap at 64 rows and DEST at 1024, so a step back is the complement.
inline void custom_mm_configure_full_in0_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Next half: the lower 8 rows of the same in0 and output faces

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 64 - 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1); // Next column: the next in1 and output face, back to the top half of the in0 face

    addr_mod_t {
        .srca = {.incr = 64 - 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 24, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_4); // Next face row: the in0 face two on, back to the first in1 face of this k face

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 64 - 24, .clr = 0, .cr = 0},
        .dest = {.incr = 1024 - 56, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5); // Next k face: one face on in SrcA and SrcB, back to the top of the output tile
}

inline void custom_mm_configure_full_in0_mop(const std::uint32_t ct_dim)
{
    load_replay_buf(
        ckernel::math::replay_buf_offset,
        CUSTOM_MM_FULL_IN0_REPLAY_LEN,
        []
        {
            // Two k faces of eight MVMULs each; the MOP issues the very last one
            for (std::uint32_t k_face = 0; k_face < 2; k_face++)
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_0, 0 /*dst*/); // row 0, column 0, top half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_1, 0 /*dst*/); // row 0, column 0, bottom half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_0, 0 /*dst*/); // row 0, column 1, top half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_4, 0 /*dst*/); // row 0, column 1, bottom half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_0, 0 /*dst*/); // row 1, column 0, top half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_1, 0 /*dst*/); // row 1, column 0, bottom half
                TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_0, 0 /*dst*/); // row 1, column 1, top half
                if (k_face == 0)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0 /*instr_mod19*/, ADDR_MOD_5, 0 /*dst*/); // row 1, column 1, bottom half
                }
            }
        });

    // As custom_mm_configure_mop: ct_dim tiles of one k-tile on one SrcB bank; the last clears both sources and
    // returns DEST to the first tile for the next k-tile
    const std::uint32_t mvmul_base      = lltt::replay_insn(ckernel::math::replay_buf_offset, CUSTOM_MM_FULL_IN0_REPLAY_LEN);
    const std::uint32_t mvmul_end_tile  = TT_OP_MVMUL(p_setrwc::CLR_A, 0 /*instr_mod19*/, ADDR_MOD_2, 0 /*dst*/);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0 /*instr_mod19*/, ADDR_MOD_3, 0 /*dst*/);

    ckernel_template tmp = ckernel_template(1 /*outer_loop_len*/, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
}

template <bool split_acc>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    load_replay_buf(
        ckernel::math::replay_buf_offset,
        custom_mm_replay_buf_len(operandB_face_r_dim, TILE_NUM_FACES),
        [operandB_face_r_dim]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

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

/**
 * @brief Configure the math thread for custom_mm: programs address mods and the MVMUL MOP.
 *
 * @tparam transpose: Transpose in1 faces during the multiply.
 * @tparam layout: DEST layout of the output tiles, see CustomMmDestLayout. Pack with the same layout through
 *                 @ref _llk_pack_custom_mm_init_.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @param ct_dim: Number of output column tiles, 1 to 34.
 * @param in1_num_faces: Faces in an in1 (SrcA) tile: 4 for [32, 32], 2 for [32, 16] ([16, 32] with transpose),
 *                       which gives [M, 16] output tiles.
 * @note Establishes the operand-driven default Src zero-substitution state, which @ref _llk_math_custom_mm_ asserts
 *       under LLK asserts. Re-run this init after any math op that leaves the flag at keep (a copy init does).
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_custom_mm_init_, giving it the same face count.
 */
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_custom_mm_init_(
    const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1, const std::uint32_t in1_num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "custom_mm: ct_dim must be 1 to 34");
    LLK_ASSERT(custom_mm_legal_in1_num_faces(in1_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");
    custom_mm_configure_addrmod<transpose, layout>();
    if (custom_mm_full_in0_r_dim(operandB_face_r_dim))
    {
        LLK_ASSERT(
            !transpose && layout == CustomMmDestLayout::Default && in1_num_faces == TILE_NUM_FACES,
            "custom_mm: a [32, 32] in0 tile takes a [32, 32] in1, no transpose and the Default layout");
        custom_mm_configure_full_in0_addrmod();
        custom_mm_configure_full_in0_mop(ct_dim);
    }
    else if (custom_mm_narrow(in1_num_faces))
    {
        custom_mm_configure_narrow_addrmod<layout>();
        custom_mm_configure_narrow_mop(operandB_face_r_dim, ct_dim);
    }
    else
    {
        custom_mm_configure_mop<custom_mm_split_acc(layout)>(operandB_face_r_dim, ct_dim);
    }

    math::reset_counters(p_setrwc::SET_ABD_F);

    // A preceding datacopy leaves the Src zero flag at keep; MVMUL needs the operand-driven value.
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Run the custom_mm MVMUL MOP over kt_dim k-tiles into DEST.
 *
 * @tparam layout: Must match the init.
 * @tparam finalize: Merge the split accumulators at the end of the call; defaults to true for the SplitAcc layouts,
 *                  which alone allow it. Pass false on all but the last of several calls accumulating into the
 *                  same tiles.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @param dst_index: First DEST output tile, counted in custom_mm_tile_rows of the layout.
 * @param kt_dim: Number of k-tiles to accumulate, 1 or more.
 * @param ct_dim: Number of output column tiles; must match the init, whose MOP bakes it in.
 * @param in1_num_faces: Faces in an in1 tile, 4 or 2; must match the init.
 * @note Call @ref _llk_math_custom_mm_init_ first, and again after any other math op (a copy init leaves the Src
 *       zero flag at keep).
 * @note Callers that know the DEST sync mode should also assert @ref custom_mm_fits_dest.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_custom_mm_.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default, bool finalize = custom_mm_split_acc(layout)>
inline void _llk_math_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim        = 1,
    const std::uint32_t in1_num_faces = TILE_NUM_FACES,
    const std::uint32_t rt_dim        = 1)
{
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "custom_mm: Src zero-substitution flag does not hold the operand-driven value; an op between the init and "
        "this call left it at keep, so denormal Src operands would not be flushed");
    LLK_ASSERT(kt_dim >= 1, "custom_mm: kt_dim must be at least 1");
    LLK_ASSERT(custom_mm_legal_in1_num_faces(in1_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");

    const std::uint32_t replay_buf_len = custom_mm_replay_buf_len(operandB_face_r_dim, in1_num_faces);
    static_assert(!finalize || custom_mm_split_acc(layout), "custom_mm: finalize merges split partials, so it needs a SplitAcc layout");
    constexpr std::uint32_t tile_rows     = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16 ? DstTileShape::Tile16x16 : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);

    if (custom_mm_full_in0_r_dim(operandB_face_r_dim))
    {
        LLK_ASSERT(!finalize, "custom_mm: a [32, 32] in0 tile has no split accumulators to finalize");
        // One pass per in0 tile-row, into that row's ct_dim output tiles; each MOP is one k-tile of the row
        for (std::uint32_t r = 0; r < rt_dim; r++)
        {
            math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index + r * ct_dim);
            for (std::uint32_t i = 0; i < kt_dim; i++)
            {
                TTI_MOP(1, 0, 0);
            }
        }
        return;
    }
    LLK_ASSERT(rt_dim == 1, "custom_mm: rt_dim above 1 needs a [32, 32] in0 tile");
    math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);

    // Run mop kt_dim - 1 times, the last inner dim is handled separately to choose whether to run finalization or not
    for (std::uint32_t i = 0; i < kt_dim - 1; i++)
    {
        TTI_MOP(1, 0, 0);
    }

    if constexpr (finalize)
    {
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
