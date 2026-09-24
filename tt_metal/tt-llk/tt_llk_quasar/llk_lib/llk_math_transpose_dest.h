// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Sets up addr mods for transpose dest operations.
 */
inline void _llk_math_transpose_dest_addrmod_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::MAX_FACE_R_DIM},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0x3ff & -ckernel::MAX_FACE_R_DIM},
    }
        .set(ADDR_MOD_2);
}

// Bands of ELTWISE_MATH_ROWS rows needed to cover one face, and the replay entries one face costs.
// In 32-bit dest the hi16 plane (DEST_NORM) and the lo16 plane (DEST_32B_LOW) are each covered by their
// own set of bands, so a face costs twice as many entries as in 16-bit dest.
constexpr std::uint32_t TRANSPOSE_BANDS_PER_FACE = ckernel::FACE_R_DIM / ELTWISE_MATH_ROWS;

template <bool EN_32BIT_DEST>
constexpr std::uint32_t transpose_face_replay_len()
{
    return (EN_32BIT_DEST ? 4U : 2U) * TRANSPOSE_BANDS_PER_FACE;
}

/**
 * @brief Reads one face from DEST into SrcB, transposed, one FPU row band per MOVD2B.
 *
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 * @tparam SRCB_ROW_BASE: First SrcB row this face occupies
 * @tparam DEST_ROW_BASE: First DEST row this face occupies
 * @tparam LO_FINAL_ADDR_MOD: Addrmod for the final band, which carries any dest jump to the next face
 */
template <bool EN_32BIT_DEST, std::uint32_t SRCB_ROW_BASE, std::uint32_t DEST_ROW_BASE, std::uint8_t LO_FINAL_ADDR_MOD = ADDR_MOD_1>
inline void _llk_math_transpose_dest_emit_face_read_()
{
    constexpr std::uint32_t LAST_BAND = ckernel::FACE_R_DIM - ELTWISE_MATH_ROWS;

#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        const std::uint8_t addr_mod = (!EN_32BIT_DEST && row == LAST_BAND) ? LO_FINAL_ADDR_MOD : ADDR_MOD_1;
        TTI_MOVD2B(p_mov::DEST_NORM, SRCB_ROW_BASE + row, addr_mod, ckernel::arch::mov_fpu_rows, p_movd2b::TRANSPOSE_ON, DEST_ROW_BASE + row);
    }

    if constexpr (EN_32BIT_DEST)
    {
#pragma GCC unroll 4
        for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
        {
            const std::uint8_t addr_mod = (row == LAST_BAND) ? LO_FINAL_ADDR_MOD : ADDR_MOD_1;
            TTI_MOVD2B(
                p_mov::DEST_32B_LOW,
                SRCB_ROW_BASE + ckernel::FACE_R_DIM + row,
                addr_mod,
                ckernel::arch::mov_fpu_rows,
                p_movd2b::TRANSPOSE_ON,
                DEST_ROW_BASE + row);
        }
    }
}

/**
 * @brief Writes one transposed face back to DEST from SrcB, one FPU row band per MOVB2D.
 *
 * Template parameters mirror @ref _llk_math_transpose_dest_emit_face_read_.
 */
template <bool EN_32BIT_DEST, std::uint32_t SRCB_ROW_BASE, std::uint32_t DEST_ROW_BASE, std::uint8_t LO_FINAL_ADDR_MOD = ADDR_MOD_1>
inline void _llk_math_transpose_dest_emit_face_write_()
{
    constexpr std::uint32_t LAST_BAND = ckernel::FACE_R_DIM - ELTWISE_MATH_ROWS;

#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        const std::uint8_t addr_mod = (!EN_32BIT_DEST && row == LAST_BAND) ? LO_FINAL_ADDR_MOD : ADDR_MOD_1;
        TTI_MOVB2D(p_mov::DEST_NORM, SRCB_ROW_BASE + row, addr_mod, ckernel::arch::mov_fpu_rows, p_movb2d::BCAST_OFF, DEST_ROW_BASE + row);
    }

    if constexpr (EN_32BIT_DEST)
    {
#pragma GCC unroll 4
        for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
        {
            const std::uint8_t addr_mod = (row == LAST_BAND) ? LO_FINAL_ADDR_MOD : ADDR_MOD_1;
            TTI_MOVB2D(
                p_mov::DEST_32B_LOW,
                SRCB_ROW_BASE + ckernel::FACE_R_DIM + row,
                addr_mod,
                ckernel::arch::mov_fpu_rows,
                p_movb2d::BCAST_OFF,
                DEST_ROW_BASE + row);
        }
    }
}

/**
 * @brief Sets up mop config for transpose dest operations.
 *
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of the tile, not only to transpose within the faces
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_mop_config_()
{
    constexpr std::uint32_t FACE_LEN = transpose_face_replay_len<EN_32BIT_DEST>();
    constexpr std::uint32_t F        = ckernel::FACE_R_DIM;
    if constexpr (EN_32BIT_DEST)
    {
        if constexpr (TRANSPOSE_OF_FACES)
        {
            constexpr std::uint32_t replay_buf_len = 3 * FACE_LEN;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // --- Simple within face transpose (reused for F0 and F3) ---
                    _llk_math_transpose_dest_emit_face_read_<true, 0, 0>();
                    _llk_math_transpose_dest_emit_face_write_<true, 0, 0, ADDR_MOD_0>(); // dst += 16

                    // --- F1+F2 within face transpose and swap: read both faces, write back swapped ---
                    _llk_math_transpose_dest_emit_face_read_<true, 0, 0, ADDR_MOD_0>();     // dst += 16 → F2
                    _llk_math_transpose_dest_emit_face_read_<true, 2 * F, 0, ADDR_MOD_2>(); // dst -= 16 → F1

                    // Write F2^T → DEST[F1 slot], then F1^T → DEST[F2 slot]
                    _llk_math_transpose_dest_emit_face_write_<true, 2 * F, 0, ADDR_MOD_0>(); // dst += 16 → F2
                    _llk_math_transpose_dest_emit_face_write_<true, 0, 0, ADDR_MOD_0>();     // dst += 16 → F3
                });
            ckernel_template temp(1, 1, TT_OP_REPLAY(FACE_LEN, replay_buf_len - FACE_LEN, 0, 0, 0, 0));
            temp.set_start_op(TT_OP_REPLAY(0, FACE_LEN, 0, 0, 0, 0));
            temp.set_end_ops(TT_OP_REPLAY(0, FACE_LEN, 0, 0, 0, 0), TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t replay_buf_len = FACE_LEN;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Read one face from DEST → SrcB (transposed), then write it back
                    _llk_math_transpose_dest_emit_face_read_<true, 0, 0>();
                    _llk_math_transpose_dest_emit_face_write_<true, 0, 0, ADDR_MOD_0>(); // dst += 16
                });
            // Loop 4 times to transpose all 4 faces
            ckernel_template temp(1 /* mop_outer_loop */, 4 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        if constexpr (TRANSPOSE_OF_FACES)
        {
            constexpr std::uint32_t replay_buf_len = 4 * FACE_LEN;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Transpose all four faces in place, then write back with faces 1<->2 swapped.
                    _llk_math_transpose_dest_emit_face_read_<false, 0, 0>();
                    _llk_math_transpose_dest_emit_face_read_<false, F, F>();
                    _llk_math_transpose_dest_emit_face_read_<false, 2 * F, 2 * F>();
                    _llk_math_transpose_dest_emit_face_read_<false, 3 * F, 3 * F>();

                    _llk_math_transpose_dest_emit_face_write_<false, 0, 0>();
                    _llk_math_transpose_dest_emit_face_write_<false, F, 2 * F>();
                    _llk_math_transpose_dest_emit_face_write_<false, 2 * F, F>();
                    _llk_math_transpose_dest_emit_face_write_<false, 3 * F, 3 * F>();
                });

            ckernel_template temp(1 /* mop_outer_loop */, 1 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t replay_buf_len = FACE_LEN;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Read one face from DEST → SrcB (transposed), then write it back
                    _llk_math_transpose_dest_emit_face_read_<false, 0, 0>();
                    _llk_math_transpose_dest_emit_face_write_<false, 0, 0, ADDR_MOD_0>(); // dst += 16
                });
            // Loop 4 times to transpose all 4 faces
            ckernel_template temp(1 /* mop_outer_loop */, 4 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
}

/**
 * @brief Initializes a 32x32 in place transpose operation on a tile in the destination register.
 *
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of the tile, not only to transpose within the faces
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 * @note @ref _llk_math_transpose_dest_ runs the configured transpose with matching template args.
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_init_()
{
    _llk_math_transpose_dest_addrmod_();
    _llk_math_transpose_dest_mop_config_<TRANSPOSE_OF_FACES, EN_32BIT_DEST>();

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Performs a 32x32 in place transpose operation on a tile in the destination register at tile_idx.
 *
 * @param tile_idx: Tile index into the destination register
 * @note Call @ref _llk_math_transpose_dest_init_ with matching template args before this function.
 * @note On the unpack thread, @ref _llk_unpack_set_srcB_dummy_valid_ (T0) must set a dummy SrcB dvalid, since the MOVD2B reads here stall on SrcB validity.
 */
inline void _llk_math_transpose_dest_(const std::uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit. MATH drains the
    // preceding math instructions so their source-bank release has landed before SRCB_VLD tests it.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
