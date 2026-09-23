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

// low_final_addr_mod applies only to the final lo16 MOV in 32-bit mode.
template <bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_emit_face_read_(
    const std::uint32_t srcb_row_base, const std::uint32_t dest_row_base, const std::uint8_t low_final_addr_mod = ADDR_MOD_1)
{
#pragma GCC unroll 4
    for (const auto row : ckernel::math::fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        TTI_MOVD2B(p_mov::DEST_NORM, srcb_row_base + row, ADDR_MOD_1, ckernel::math::FPU_MOV_ROWS, p_movd2b::TRANSPOSE_ON, dest_row_base + row);
    }

    if constexpr (EN_32BIT_DEST)
    {
#pragma GCC unroll 4
        for (const auto row : ckernel::math::fpu_row_offsets<ckernel::FACE_R_DIM>())
        {
            const std::uint8_t addr_mod = (row + ELTWISE_MATH_ROWS == ckernel::FACE_R_DIM) ? low_final_addr_mod : ADDR_MOD_1;
            TTI_MOVD2B(
                p_mov::DEST_32B_LOW,
                srcb_row_base + ckernel::FACE_R_DIM + row,
                addr_mod,
                ckernel::math::FPU_MOV_ROWS,
                p_movd2b::TRANSPOSE_ON,
                dest_row_base + row);
        }
    }
}

// final_addr_mod applies only after both planes have been written in 32-bit mode.
template <bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_emit_face_write_(
    const std::uint32_t srcb_row_base, const std::uint32_t dest_row_base, const std::uint8_t final_addr_mod = ADDR_MOD_1)
{
#pragma GCC unroll 4
    for (const auto row : ckernel::math::fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        const std::uint8_t addr_mod = (!EN_32BIT_DEST && row + ELTWISE_MATH_ROWS == ckernel::FACE_R_DIM) ? final_addr_mod : ADDR_MOD_1;
        TTI_MOVB2D(p_mov::DEST_NORM, srcb_row_base + row, addr_mod, ckernel::math::FPU_MOV_ROWS, p_movb2d::BCAST_OFF, dest_row_base + row);
    }

    if constexpr (EN_32BIT_DEST)
    {
#pragma GCC unroll 4
        for (const auto row : ckernel::math::fpu_row_offsets<ckernel::FACE_R_DIM>())
        {
            const std::uint8_t addr_mod = (row + ELTWISE_MATH_ROWS == ckernel::FACE_R_DIM) ? final_addr_mod : ADDR_MOD_1;
            TTI_MOVB2D(
                p_mov::DEST_32B_LOW,
                srcb_row_base + ckernel::FACE_R_DIM + row,
                addr_mod,
                ckernel::math::FPU_MOV_ROWS,
                p_movb2d::BCAST_OFF,
                dest_row_base + row);
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
    static_assert(ELTWISE_MATH_ROWS == 8 || ELTWISE_MATH_ROWS == 4, "transpose dest supports MATH_ROWS of 8 (Quasar) or 4 (4row_arch)");

    constexpr std::uint32_t face_moves      = ckernel::FACE_R_DIM / ELTWISE_MATH_ROWS;
    constexpr std::uint32_t face_replay_len = EN_32BIT_DEST ? (4 * face_moves) : (2 * face_moves);

    if constexpr (!TRANSPOSE_OF_FACES)
    {
        load_replay_buf<0, face_replay_len>(
            []
            {
                _llk_math_transpose_dest_emit_face_read_<EN_32BIT_DEST>(0, 0);
                _llk_math_transpose_dest_emit_face_write_<EN_32BIT_DEST>(0, 0, ADDR_MOD_0);
            });

        ckernel_template temp(1, 4, TT_OP_REPLAY(0, face_replay_len, 0, 0, 0, 0));
        temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else if constexpr (EN_32BIT_DEST)
    {
        constexpr std::uint32_t replay_buf_len = 3 * face_replay_len;
        load_replay_buf<0, replay_buf_len>(
            []
            {
                // F0 face transpose, also replayed for F3.
                _llk_math_transpose_dest_emit_face_read_<true>(0, 0);
                _llk_math_transpose_dest_emit_face_write_<true>(0, 0, ADDR_MOD_0);

                // Read F1/F2, then write them back swapped.
                _llk_math_transpose_dest_emit_face_read_<true>(0, 0, ADDR_MOD_0);
                _llk_math_transpose_dest_emit_face_read_<true>(32, 0, ADDR_MOD_2);
                _llk_math_transpose_dest_emit_face_write_<true>(32, 0, ADDR_MOD_0);
                _llk_math_transpose_dest_emit_face_write_<true>(0, 0, ADDR_MOD_0);
            });

        ckernel_template temp(1, 1, TT_OP_REPLAY(face_replay_len, replay_buf_len - face_replay_len, 0, 0, 0, 0));
        temp.set_start_op(TT_OP_REPLAY(0, face_replay_len, 0, 0, 0, 0));
        temp.set_end_ops(TT_OP_REPLAY(0, face_replay_len, 0, 0, 0, 0), TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        constexpr std::uint32_t replay_buf_len = 4 * face_replay_len;
        load_replay_buf<0, replay_buf_len>(
            []
            {
                _llk_math_transpose_dest_emit_face_read_<false>(0, 0);
                _llk_math_transpose_dest_emit_face_read_<false>(16, 16);
                _llk_math_transpose_dest_emit_face_read_<false>(32, 32);
                _llk_math_transpose_dest_emit_face_read_<false>(48, 48);

                constexpr std::uint32_t first_swap_src  = ELTWISE_MATH_ROWS == 8 ? 16 : 32;
                constexpr std::uint32_t second_swap_src = ELTWISE_MATH_ROWS == 8 ? 32 : 16;
                _llk_math_transpose_dest_emit_face_write_<false>(0, 0);
                _llk_math_transpose_dest_emit_face_write_<false>(first_swap_src, second_swap_src);
                _llk_math_transpose_dest_emit_face_write_<false>(second_swap_src, first_swap_src);
                _llk_math_transpose_dest_emit_face_write_<false>(48, 48);
            });

        ckernel_template temp(1, 1, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
        temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
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
