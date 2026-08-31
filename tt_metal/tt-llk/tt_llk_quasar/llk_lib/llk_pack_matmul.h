// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Builds the MOP for packing a matmul output subblock via Packer 0.
 *
 * Packer 1 (SrcS) uses llk_srcs.h.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param subblock_r_dim: Number of tiles in the row dimension of a matrix block.
 * @param subblock_c_dim: Number of tiles in the column dimension of a matrix block.
 * @param num_subblocks_c_dim: Number of subblocks in the column dimension of a matrix block.
 * @param tensor_shape: Shape of each output tile.
 */
inline void _llk_pack_matmul_mop_config_(
    const std::uint32_t buf_desc_id,
    const std::uint32_t subblock_r_dim,
    const std::uint32_t subblock_c_dim,
    const std::uint32_t num_subblocks_c_dim,
    const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    const std::uint32_t MOP_OUTER_LOOP = subblock_r_dim;
    const std::uint32_t num_faces           = tensor_shape.total_num_faces();
    const bool full_tile                    = num_faces == NUM_FACES;
    const std::uint32_t pack_units_per_tile = full_tile ? 1 : num_faces;
    const std::uint32_t MOP_INNER_LOOP      = subblock_c_dim * pack_units_per_tile;
    const std::uint32_t pack_instrn         = TT_OP_PACR0_TILE_INC(1 /* Dst (L1) tile idx */, full_tile ? 1 : 0 /* Src tile idx */, buf_desc_id, 0);
    const std::uint32_t incr_l1_ptr =
        TT_OP_INC_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, (subblock_c_dim * num_subblocks_c_dim - subblock_c_dim) * pack_units_per_tile);
    const std::uint32_t src_face_stride = full_tile || tensor_shape.face_r_dim >= MAX_FPU_ROWS ? 1 : MAX_FPU_ROWS / tensor_shape.face_r_dim;
    const std::uint32_t incr_src_ptr    = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, src_face_stride);
    ckernel_template temp =
        full_tile ? ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn) : ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, incr_src_ptr);
    temp.set_end_op(incr_l1_ptr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the packer for packing a matmul output subblock via Packer 0.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param subblock_r_dim: Number of tiles in the row dimension of a matrix block.
 * @param subblock_c_dim: Number of tiles in the column dimension of a matrix block.
 * @param num_subblocks_c_dim: Number of subblocks in the column dimension of a matrix block.
 * @param tensor_shape: Shape of each output tile.
 * @note @ref _llk_pack_matmul_ is the matching execute call on this thread.
 */
inline void _llk_pack_matmul_init_(
    const std::uint32_t buf_desc_id,
    const std::uint32_t subblock_r_dim,
    const std::uint32_t subblock_c_dim,
    const std::uint32_t num_subblocks_c_dim,
    const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    _llk_pack_matmul_mop_config_(buf_desc_id, subblock_r_dim, subblock_c_dim, num_subblocks_c_dim, tensor_shape);
}

/**
 * @brief Packs out tiles via Packer 0 using the matmul MOP.
 *
 * @param start_math_dest_tile_idx: The tile index into the math destination register that the packer starts packing from.
 * @param start_l1_tile_idx: The tile index into the L1 output buffer that the packer starts packing into.
 * @param tensor_shape: Shape of each output tile.
 * @note Call @ref _llk_pack_matmul_init_ with matching template args before this function.
 */
inline void _llk_pack_matmul_(
    const std::uint32_t start_math_dest_tile_idx, const std::uint32_t start_l1_tile_idx, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    //(TODO) RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users offset idx

    const std::uint32_t num_faces            = tensor_shape.total_num_faces();
    const bool full_tile                     = num_faces == NUM_FACES;
    const std::uint32_t src_face_stride      = full_tile || tensor_shape.face_r_dim >= MAX_FPU_ROWS ? 1 : MAX_FPU_ROWS / tensor_shape.face_r_dim;
    const std::uint32_t math_dest_tile_scale = full_tile ? 1 : num_faces * src_face_stride;
    const std::uint32_t l1_tile_scale        = full_tile ? 1 : num_faces;

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, start_math_dest_tile_idx * math_dest_tile_scale);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, start_l1_tile_idx * l1_tile_scale);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
