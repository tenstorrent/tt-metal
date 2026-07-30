// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// DEST scratch slot for the second input face-row (F2&F3) partial.
static constexpr std::uint32_t REDUCE_BLOCK_SLOT1_DST = TILE_R_DIM;

/**
 * @brief Program the address modifier used by the block reduce_max_row pool + transpose.
 *
 * ADDR_MOD_0 is the only one used: no counter movement (the pool parks DEST at a fixed slot; MOVD2B
 * reads DEST at the counter and MOVB2D writes the transposed column back at the counter).
 */
inline void reduce_block_max_row_configure_addrmod()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
}

/**
 * @brief Transpose one pooled 1x16 row partial (at the current DEST counter) into a 16x1 column.
 */
inline void reduce_block_max_row_transpose_face_row(const bool wide_face)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    // Move the row partial into SrcB rows [16-31], transposed into rows [32-47].
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    // Write the transposed column back to DEST via plain MOVB2D (SrcB rows 32-47 -> DEST). First move
    // covers DEST rows 0-7.
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
    if (wide_face)
    {
        // face_r_dim > ELTWISE_MATH_ROWS (full 16-row face): second move covers DEST rows 8-15.
        TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
    }
}

/**
 * @brief Runtime-block_ct_dim MOP config for block reduce_max_row.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_runtime_(const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");

    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);
    // dvalid-streaming pool. The unpacker writes every face to SrcA rows 0-15 and flips the write bank
    // per face (see the unpacker's Dst_Face_Idx_Inc=0), so the two SrcA banks act as a double-buffer.
    // Each GMPOOL reads rows 0-15 of the current read bank (ADDR_MOD_0 = no address advance) and
    // CLR_SRCA_VLD clears that bank's dvalid -- which frees it for the unpacker to refill AND rotates the
    // read bank to the next face. The four faces stream F0->F1->F2->F3 through the two rotating banks at
    // a fixed address; the dst immediate routes F0,F1 -> slot0 and F2,F3 -> slot1.
    // NOTE: GMPOOL consumes SrcA by bank rotation, NOT by an srca offset (that is the matmul/MVMUL
    // model). An address-walk (Dst_Face_Idx_Inc=1 to rows 0/16/32/48 + srca+=16) was tried and fails --
    // the faces scatter across both banks and the walk reads never-written rows.
    const std::uint32_t pool_len = (two_face_rows ? 4u : 2u);

    load_replay_buf(
        0,
        pool_len,
        false,
        0,
        0,
        [two_face_rows]
        {
            // Every face is pooled with CLR_SRCA_VLD (consume + rotate the SrcA read bank, releasing the
            // consumed bank for the unpacker). The LAST GMPOOL's CLR_SRCA_VLD doubles as the tile's single
            // SrcA release, so there is deliberately NO trailing SETRWC(CLR_A).
            // RULE (device-verified): exactly ONE SrcA release per tile. A terminal CLR_SRCA_VLD *and* a
            // SETRWC(CLR_A) both release -> double-release -> DEST is zeroed. Either one alone -> correct.
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0); // F0 -> slot0
            if (two_face_rows)
            {
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);                      // F1 -> slot0
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_BLOCK_SLOT1_DST); // F2 -> slot1
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_BLOCK_SLOT1_DST); // F3 -> slot1 (terminal release)
            }
            else
            {
                // Tiny tile (single face-row): only F0,F1 -> slot0; F1 is the terminal SrcA release.
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0); // F1 -> slot0 (terminal release)
            }
        });

    const std::uint32_t pool_replay = TT_OP_REPLAY(0, pool_len, 0, 0, 0, 0);
    ckernel_template temp(1 /*outer*/, block_ct_dim /*inner*/, pool_replay);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Runtime-block_ct_dim init for block reduce_max_row.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_runtime_(const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    reduce_block_max_row_configure_addrmod();
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));
    _reset_counters_<p_setrwc::SET_ABD_F>();
    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_runtime_()
{
}

/**
 * @brief Runtime-block_ct_dim execute for block reduce_max_row. The block tile count is baked into
 *        the MOP by init, so only dst_index / tensor_shape are needed here.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_runtime_(const std::uint32_t dst_index, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");

    _set_dst_write_addr_by_rows_(dst_index);

    // POOL PHASE: run the MOP to stream every face through the rotating SrcA banks, pooling
    // F0,F1 -> slot0 and F2,F3 -> slot1.
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // TRANSPOSE PHASE: transpose each pooled row partial into a column (once).
    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);
    const bool wide_face     = (tensor_shape.face_r_dim > ELTWISE_MATH_ROWS);

    reduce_block_max_row_transpose_face_row(wide_face);
    if (two_face_rows)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, TILE_R_DIM, p_setrwc::SET_D);
        reduce_block_max_row_transpose_face_row(wide_face);
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
