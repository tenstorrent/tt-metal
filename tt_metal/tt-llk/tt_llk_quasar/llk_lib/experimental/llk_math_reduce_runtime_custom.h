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
 * @brief Transpose one pooled 1xN row partial (at the current DEST counter) into an Nx1 column.
 *
 * @param face_r_dim Number of rows in the face = height of the transposed output column. The column is
 *                   written back to DEST with MOVB2D moves of ELTWISE_MATH_ROWS rows each,
 *                   so ceil(face_r_dim / ELTWISE_MATH_ROWS) moves are emitted -- one per
 *                   ELTWISE_MATH_ROWS-row band the face actually spans.
 */
inline void reduce_block_max_row_transpose_face_row(const std::uint32_t face_r_dim)
{
    // Move the 1xN row partial into SrcB rows [16-31] and transpose it into the Nx1 column at SrcB rows
    // [32-47] (paired transpose-on / transpose-off MOVD2B, mirroring native reduce-row).
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

    // Write the transposed column back to DEST (SrcB rows 32-47 -> DEST rows 0..face_r_dim-1).
    if constexpr (ELTWISE_MATH_ROWS == 8)
    {
        // Quasar: 8-row MOVB2D. A 16-row face needs 2 moves; an <=8-row face needs 1.
        TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0); // rows 0-7
        if (face_r_dim > 8)
        {
            TTI_MOVB2D(
                p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8); // rows 8-15
        }
    }
    else if constexpr (ELTWISE_MATH_ROWS == 4)
    {
        // (4-row FPU): 4-row MOVB2D -> up to 4 moves for a 16-row face.
        TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 0); // rows 0-3
        if (face_r_dim > 4)
        {
            TTI_MOVB2D(
                p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 4, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 4); // rows 4-7
        }
        if (face_r_dim > 8)
        {
            TTI_MOVB2D(
                p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 8); // rows 8-11
        }
        if (face_r_dim > 12)
        {
            TTI_MOVB2D(
                p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 12, ADDR_MOD_0, p_mov_src_to_dest::MOV_4_ROWS, p_movb2d::BCAST_OFF, 12); // rows 12-15
        }
    }
}

/**
 * @brief Runtime-block_ct_dim MOP config for block reduce_max_row.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_runtime_(const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    // Using static_assert here for compile-time check.
    static_assert(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");
    LLK_ASSERT(
        tensor_shape.num_faces_c_dim == 2, "block reduce_max_row requires a 32-wide operand (num_faces_c_dim == 2); narrow tiles (32x16, 16x16) unsupported");

    // A face_row is a term used for two faces in a 16x32 partial tile (F0&F1 or F2&F3).
    // The pool phase of the block reduce_max_row MOP only pools the two face_rows into two fixed DEST slots (slot0 for F0&F1, slot1 for F2&F3).
    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);

    const std::uint32_t pool_len = (two_face_rows ? 4u : 2u);

    load_replay_buf(
        0,
        pool_len,
        false,
        0,
        0,
        [two_face_rows]
        {
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
    static_assert(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");

    _set_dst_write_addr_by_rows_(dst_index);

    // POOL PHASE: run the MOP to stream every face through the rotating SrcA banks, pooling
    // F0,F1 -> slot0 and F2,F3 -> slot1.
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // TRANSPOSE PHASE: transpose each pooled row partial into a column (once).
    // A face_row is a term used for two faces in a 16x32 partial tile (F0&F1 or F2&F3).
    // The pool phase of the block reduce_max_row MOP only pools the two face_rows into two fixed DEST slots (slot0 for F0&F1, slot1 for F2&F3).
    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);

    reduce_block_max_row_transpose_face_row(tensor_shape.face_r_dim);
    if (two_face_rows)
    {
        // Advance the DEST counter to slot1 (row TILE_R_DIM = 32, where the pool wrote the F2&F3 partial).
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, TILE_R_DIM, p_setrwc::SET_D);
        reduce_block_max_row_transpose_face_row(tensor_shape.face_r_dim);
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
