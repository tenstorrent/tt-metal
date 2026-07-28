// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "experimental/llk_math_reduce_custom.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// Runtime-block_ct_dim variant of the block reduce_max_row math kernel. Same algorithm as the
// compile-time header (accumulate the row-max across the block, then transpose once); only the block
// tile count is a runtime argument. Shared helpers (addrmod, transpose, slot layout) come from
// llk_math_reduce_custom.h. See that header for the full algorithm description.

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
    const std::uint32_t pool_len = (two_face_rows ? 5u : 3u);

    load_replay_buf(
        0,
        pool_len,
        false,
        0,
        0,
        [two_face_rows]
        {
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0); // F0 -> slot0, consume + rotate bank
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0); // F1 -> slot0, consume + rotate bank
            if (two_face_rows)
            {
                TTI_GMPOOL(
                    p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_BLOCK_SLOT1_DST); // F2 -> slot1, consume + rotate bank
                TTI_GMPOOL(
                    p_gpool::CLR_NONE,
                    p_gpool::DIM_16X16,
                    ADDR_MOD_0,
                    p_gpool::INDEX_DIS,
                    REDUCE_BLOCK_SLOT1_DST); // F3 -> slot1, keep valid (last face, nothing to hand off to)
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB);
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
