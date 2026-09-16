// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

static_assert(ELTWISE_MATH_ROWS == 4, "4row_quasar overrides require ELTWISE_MATH_ROWS == 4");

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

// The 4-row FPU moves 4 rows per MOV, so a 16-row face needs 4 MOVs instead of
// two. In 32-bit dest mode the hi16 plane (DEST_NORM) and lo16 plane
// (DEST_32B_LOW) are each covered by their own set of 4-row MOVs.
template <bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_emit_face_read_4row_(
    const std::uint32_t srcb_row_base, const std::uint32_t dest_row_base, const std::uint8_t low_final_addr_mod = ADDR_MOD_1)
{
    constexpr std::uint32_t rows_per_instr = 4;
    constexpr std::uint32_t face_rows      = ckernel::FACE_R_DIM;

    if constexpr (EN_32BIT_DEST)
    {
        // Read hi16 plane from DEST → SrcB (transposed)
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            TTI_MOVD2B(p_mov::DEST_NORM, srcb_row_base + row, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, p_movd2b::TRANSPOSE_ON, dest_row_base + row);
        }

        // Read lo16 plane from DEST → SrcB (transposed)
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            const std::uint8_t addr_mod = (row + rows_per_instr == face_rows) ? low_final_addr_mod : ADDR_MOD_1;
            TTI_MOVD2B(p_mov::DEST_32B_LOW, srcb_row_base + face_rows + row, addr_mod, p_movd2b::MOV_4_ROWS, p_movd2b::TRANSPOSE_ON, dest_row_base + row);
        }
    }
    else
    {
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            TTI_MOVD2B(p_mov::DEST_NORM, srcb_row_base + row, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, p_movd2b::TRANSPOSE_ON, dest_row_base + row);
        }
    }
}

template <bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_emit_face_write_4row_(
    const std::uint32_t srcb_row_base, const std::uint32_t dest_row_base, const std::uint8_t low_final_addr_mod = ADDR_MOD_1)
{
    constexpr std::uint32_t rows_per_instr = 4;
    constexpr std::uint32_t face_rows      = ckernel::FACE_R_DIM;

    if constexpr (EN_32BIT_DEST)
    {
        // Write hi16 plane back to DEST from SrcB
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            TTI_MOVB2D(p_mov::DEST_NORM, srcb_row_base + row, ADDR_MOD_1, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_row_base + row);
        }

        // Write lo16 plane back to DEST from SrcB
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            const std::uint8_t addr_mod = (row + rows_per_instr == face_rows) ? low_final_addr_mod : ADDR_MOD_1;
            TTI_MOVB2D(p_mov::DEST_32B_LOW, srcb_row_base + face_rows + row, addr_mod, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_row_base + row);
        }
    }
    else
    {
        for (std::uint32_t row = 0; row < face_rows; row += rows_per_instr)
        {
            const std::uint8_t addr_mod = (row + rows_per_instr == face_rows) ? low_final_addr_mod : ADDR_MOD_1;
            TTI_MOVB2D(p_mov::DEST_NORM, srcb_row_base + row, addr_mod, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, dest_row_base + row);
        }
    }
}

/**
 * @brief Sets up mop config for transpose dest operations.
 *
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of the tile, not only to transpose within the faces
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 * @note The FPU moves four rows per MOV, so four MOVs cover each 16-row face plane.
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_mop_config_()
{
    constexpr std::uint32_t rows_per_instr  = 4;
    constexpr std::uint32_t face_ops        = ckernel::FACE_R_DIM / rows_per_instr;
    constexpr std::uint32_t face_replay_len = EN_32BIT_DEST ? (4 * face_ops) : (2 * face_ops);

    if constexpr (EN_32BIT_DEST)
    {
        if constexpr (TRANSPOSE_OF_FACES)
        {
            constexpr std::uint32_t replay_buf_len = 3 * face_replay_len;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // F0 face transpose
                    _llk_math_transpose_dest_emit_face_read_4row_<true>(0, 0);
                    _llk_math_transpose_dest_emit_face_write_4row_<true>(0, 0, ADDR_MOD_0);

                    // F1/F2 transpose + swap
                    _llk_math_transpose_dest_emit_face_read_4row_<true>(0, 0, ADDR_MOD_0);
                    _llk_math_transpose_dest_emit_face_read_4row_<true>(32, 0, ADDR_MOD_2);
                    _llk_math_transpose_dest_emit_face_write_4row_<true>(32, 0, ADDR_MOD_0);
                    _llk_math_transpose_dest_emit_face_write_4row_<true>(0, 0, ADDR_MOD_0);
                });

            ckernel_template temp(1, 1, TT_OP_REPLAY(face_replay_len, replay_buf_len - face_replay_len, 0, 0, 0, 0));
            temp.set_start_op(TT_OP_REPLAY(0, face_replay_len, 0, 0, 0, 0));
            temp.set_end_ops(TT_OP_REPLAY(0, face_replay_len, 0, 0, 0, 0), TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t replay_buf_len = face_replay_len;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    _llk_math_transpose_dest_emit_face_read_4row_<true>(0, 0);
                    _llk_math_transpose_dest_emit_face_write_4row_<true>(0, 0, ADDR_MOD_0);
                });

            ckernel_template temp(1, 4, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        if constexpr (TRANSPOSE_OF_FACES)
        {
            constexpr std::uint32_t replay_buf_len = 8 * face_ops;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    _llk_math_transpose_dest_emit_face_read_4row_<false>(0, 0);
                    _llk_math_transpose_dest_emit_face_read_4row_<false>(16, 16);
                    _llk_math_transpose_dest_emit_face_read_4row_<false>(32, 32);
                    _llk_math_transpose_dest_emit_face_read_4row_<false>(48, 48);

                    _llk_math_transpose_dest_emit_face_write_4row_<false>(0, 0);
                    _llk_math_transpose_dest_emit_face_write_4row_<false>(32, 16);
                    _llk_math_transpose_dest_emit_face_write_4row_<false>(16, 32);
                    _llk_math_transpose_dest_emit_face_write_4row_<false>(48, 48);
                });

            ckernel_template temp(1, 1, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t replay_buf_len = face_replay_len;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    _llk_math_transpose_dest_emit_face_read_4row_<false>(0, 0);
                    _llk_math_transpose_dest_emit_face_write_4row_<false>(0, 0, ADDR_MOD_0);
                });

            ckernel_template temp(1, 4, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
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
