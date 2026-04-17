// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Sets up addr mods for transpose dest operations
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

/**
 * @brief Sets up mop config for transpose dest operations
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of the tile, not only to transpose within the faces
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_mop_config_()
{
    if (EN_32BIT_DEST)
    {
        if (TRANSPOSE_OF_FACES)
        {
            constexpr std::uint32_t replay_buf_len = 24;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // --- Instructions[0..7]: Simple within face transpose (reused for F0 and F3) ---

                    // Read hi16 from DEST → SrcB[0:15] (transposed)
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);
                    // Read lo16 from DEST → SrcB[16:31] (transposed)
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 24, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);

                    // Write hi16 to DEST from SrcB[0:15]
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
                    // Write lo16 to DEST from SrcB[16:31]
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8); // dst += 16

                    // --- Instructions[8..23]: F1+F2 within face transpose and swap: read both faces, write back swapped ---

                    // Read F1 hi16 → SrcB[0:15] (transposed)
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);
                    // Read F1 lo16 → SrcB[16:31] (transposed)
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 24, ADDR_MOD_0, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8); // dst += 16 → F2

                    // Read F2 hi16 → SrcB[32:47] (transposed)
                    TTI_MOVD2B(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);
                    // Read F2 lo16 → SrcB[48:63] (transposed)
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 48, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 56, ADDR_MOD_2, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8); // dst -= 16 → F1

                    // Write F2^T SrcB[48:63] → DEST[F1 slot]
                    TTI_MOVB2D(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 48, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 56, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8); // dst += 16 → F2

                    // Write F1^T SrcB[0..31] → DEST[F2 slot]
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8); // dst += 16 → F3
                });
            ckernel_template temp(1, 1, TT_OP_REPLAY(8, replay_buf_len - 8, 0, 0, 0, 0));
            temp.set_start_op(TT_OP_REPLAY(0, 8, 0, 0, 0, 0));
            temp.set_end_ops(TT_OP_REPLAY(0, 8, 0, 0, 0, 0), TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCAB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t replay_buf_len = 8;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // --- Instructions[0..3]: Read one face from DEST → SrcB (transposed) ---
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);

                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 24, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);

                    // --- Instructions[4..7]: Write one transposed face to DEST from SrcB ---
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);

                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8); // dst += 16
                });
            // Loop 4 times to transpose all 4 faces
            ckernel_template temp(1 /* mop_outer_loop */, 4 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCAB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        constexpr std::uint32_t replay_buf_len = 16;
        load_replay_buf<0, replay_buf_len>(
            []
            {   // --- Instructions[0..7]: Read from DEST → SrcB (transposed) ---
                // Transpose all 8 half-faces in place, then write back with faces 1<->2 swapped.
                TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 0);
                TTI_MOVD2B(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 8);
                TTI_MOVD2B(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 16);
                TTI_MOVD2B(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 24);
                TTI_MOVD2B(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 32);
                TTI_MOVD2B(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 40);
                TTI_MOVD2B(p_mov::DEST_NORM, 48, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 48);
                TTI_MOVD2B(p_mov::DEST_NORM, 56, ADDR_MOD_1, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_ON, 56);

                // --- Instructions[8..15]: Write to DEST from SrcB with faces 1<->2 swapped ---
                TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
                TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
                TTI_MOVB2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 32);
                TTI_MOVB2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 40);
                TTI_MOVB2D(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 16);
                TTI_MOVB2D(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 24);
                TTI_MOVB2D(p_mov::DEST_NORM, 48, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 48);
                TTI_MOVB2D(p_mov::DEST_NORM, 56, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 56);
            });

        ckernel_template temp(1 /* mop_outer_loop */, 1 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
        temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCAB_VLD, 0, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes a 32x32 in place transpose operation on a tile in the destination register.
 * @tparam TRANSPOSE_OF_FACES: Set to true to transpose the faces of the tile, not only to transpose within the faces
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit mode
 */
template <bool TRANSPOSE_OF_FACES, bool EN_32BIT_DEST>
inline void _llk_math_transpose_dest_init_()
{
    static_assert(TRANSPOSE_OF_FACES || EN_32BIT_DEST, "Within-face-only transpose is not supported for 16-bit destination register");

    _llk_math_transpose_dest_addrmod_();
    _llk_math_transpose_dest_mop_config_<TRANSPOSE_OF_FACES, EN_32BIT_DEST>();

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Performs a 32x32 in place transpose operation on a tile in the destination register at tile_idx.
 * @param tile_idx: Tile index into the destination register
 */
inline void _llk_math_transpose_dest_(const std::uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    // Wait condition SRCA_VLD is required as MOVB2A doesn't automatically wait
    // for SrcA[MatrixUnit.SrcABank].AllowedClient == SrcClient::MatrixUnit.
    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1, p_stall::SRCB_VLD, p_stall::SRCA_VLD);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
