// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief Builds the MOP that unpacks operand 0 into SrcB and operand 1 into SrcA for matrix multiply.
 *
 * The matrix multiply FPU operation computes SrcB * SrcA. To obtain the row-major result
 * Input0 * Input1, SrcA and SrcB are loaded from Input1 and Input0 respectively. This unpacker only
 * sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]; kt_dim is assumed to be iterated over outside this
 * call. Constraints: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 *
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 32
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 */
inline void _llk_unpack_matmul_mop_config_(
    std::uint32_t buf_desc_id_0, std::uint32_t buf_desc_id_1, std::uint8_t ct_dim, std::uint8_t rt_dim, std::uint32_t kt_dim)
{
    const bool reuse_a                     = ct_dim >= rt_dim;
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = reuse_a ? ct_dim : rt_dim;
    std::uint32_t unpack_instrn;
    // static uint inc_l1_instrn;
    std::uint32_t unpack_reuse_instrn;

    if (reuse_a)
    {
        unpack_instrn = TT_OP_UNPACR0_TILE_INC(0, 1, buf_desc_id_1, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 1);
        unpack_reuse_instrn = TT_OP_UNPACR1_TILE_INC(0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_instrn = TT_OP_UNPACR1_TILE_INC(0, kt_dim, buf_desc_id_0, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, KT_DIM);
        unpack_reuse_instrn = TT_OP_UNPACR0_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn /*, inc_l1_instrn*/);
    temp.set_start_op(unpack_reuse_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Returns unpacker source-counter units per software tile.
 *
 * Full four-face tiles use one hardware-tile unit; other supported shapes use face units.
 */
inline std::uint32_t _llk_unpack_matmul_src_tile_scale_(const TensorShape& tensor_shape)
{
    return tensor_shape.total_num_faces() == MAX_NUM_FACES ? 1 : tensor_shape.total_num_faces();
}

/**
 * @brief Records one operand's face-by-face unpack sequence.
 *
 * Faces of at most eight rows use 8-row Src offsets. Shorter faces clear the current Src bank before unpack.
 *
 * @tparam UNP_SEL: Destination Src register, values = <p_unpacr::UNP_A/UNP_B>
 * @param replay_start: Replay-buffer index at which to start recording.
 * @param buf_desc_id: Buffer descriptor ID in the unpacker BFD table.
 * @param tensor_shape: Shape of the operand tile.
 * @return Number of instructions recorded.
 */
template <std::uint32_t UNP_SEL>
inline std::uint32_t _llk_unpack_matmul_load_face_replay_(const std::uint32_t replay_start, const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    static_assert(UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_B, "Matmul operands must unpack to SrcA or SrcB");

    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const bool pad_src                  = tensor_shape.face_r_dim < MAX_FPU_ROWS;
    const bool use_eight_row_offsets    = tensor_shape.face_r_dim <= MAX_FPU_ROWS;
    const std::uint32_t replay_len      = 1 + pad_src + num_faces * (use_eight_row_offsets ? 2 : 1);
    const std::uint32_t dst_tile_stride = use_eight_row_offsets ? (MAX_FPU_ROWS / tensor_shape.face_r_dim) : 1;

    load_replay_buf(
        replay_start,
        replay_len,
        false /*exec_while_loading*/,
        0 /*set_mutex*/,
        0 /*last*/,
        [buf_desc_id, num_faces, pad_src, use_eight_row_offsets, dst_tile_stride]
        {
            TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
            if (pad_src)
            {
                TTI_UNPACR_NOP(
                    UNP_SEL, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear current bank */, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);
            }

            for (std::uint32_t face = 0; face < num_faces; face++)
            {
                if (use_eight_row_offsets)
                {
                    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, face * dst_tile_stride);
                }
                const std::uint32_t dst_face   = use_eight_row_offsets ? 0 : face;
                const std::uint32_t set_dvalid = face + 1 == num_faces ? 1 : 0;
                if constexpr (UNP_SEL == p_unpacr::UNP_A)
                {
                    TT_UNPACR0_FACE(dst_face, face, 0, 0, buf_desc_id, set_dvalid);
                }
                else
                {
                    TT_UNPACR1_FACE(dst_face, face, 0, 0, buf_desc_id, set_dvalid);
                }
            }
        });
    return replay_len;
}

/**
 * @brief Builds the face-by-face MOP used when either matmul operand is not a full four-face tile.
 *
 * Operand mapping and reuse selection match @ref _llk_unpack_matmul_mop_config_.
 *
 * @param buf_desc_id_0/1: Buffer descriptor IDs for Input 0/SrcB and Input 1/SrcA.
 * @param ct_dim: Number of Input 1 tiles along the output-column dimension.
 * @param rt_dim: Number of Input 0 tiles along the output-row dimension.
 * @param kt_dim: Number of tiles along the contraction dimension.
 * @param src_b_shape: Input 0/SrcB tile shape.
 * @param src_a_shape: Input 1/SrcA tile shape.
 */
inline void _llk_unpack_matmul_face_mop_config_(
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const std::uint8_t ct_dim,
    const std::uint8_t rt_dim,
    const std::uint32_t kt_dim,
    const TensorShape& src_b_shape,
    const TensorShape& src_a_shape)
{
    const bool reuse_a                    = ct_dim >= rt_dim;
    const std::uint32_t reuse_replay_len  = reuse_a ? _llk_unpack_matmul_load_face_replay_<p_unpacr::UNP_B>(0, buf_desc_id_0, src_b_shape)
                                                    : _llk_unpack_matmul_load_face_replay_<p_unpacr::UNP_A>(0, buf_desc_id_1, src_a_shape);
    const std::uint32_t stream_replay_len = reuse_a ? _llk_unpack_matmul_load_face_replay_<p_unpacr::UNP_A>(reuse_replay_len, buf_desc_id_1, src_a_shape)
                                                    : _llk_unpack_matmul_load_face_replay_<p_unpacr::UNP_B>(reuse_replay_len, buf_desc_id_0, src_b_shape);
    const std::uint32_t stream_tile_inc = reuse_a ? _llk_unpack_matmul_src_tile_scale_(src_a_shape) : _llk_unpack_matmul_src_tile_scale_(src_b_shape) * kt_dim;
    const std::uint32_t stream_inc_instrn = reuse_a ? TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, stream_tile_inc)
                                                    : TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, stream_tile_inc);

    ckernel_template temp(1, reuse_a ? ct_dim : rt_dim, TT_OP_REPLAY(reuse_replay_len, stream_replay_len, 0, 0, 0, 0), stream_inc_instrn);
    temp.set_start_op(TT_OP_REPLAY(0, reuse_replay_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack operand 0 into SrcB and operand 1 into SrcA for matrix multiply.
 *
 * The matrix multiply FPU operation computes SrcB * SrcA. To obtain the row-major result
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim], SrcA and SrcB are loaded
 * from Input1 and Input0 respectively. This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim];
 * kt_dim is assumed to be iterated over outside this call. Constraints: ct_dim * rt_dim <= 8 tiles in
 * a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * When both operands are full four-face tiles, the unpacker uses the TILE_INC MOP; otherwise it unpacks face-by-face.
 *
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, currently only supported for SrcA but can support other unpackers, values = <true/false>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 * @param src_b_shape: Input 0/SrcB tile shape. Default is a full 32x32 tile.
 * @param src_a_shape: Input 1/SrcA tile shape. Default is a full 32x32 tile.
 * @note On the math thread, pair with @ref _llk_math_matmul_init_ (T1); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_matmul_ is the matching execute call on this thread.
 */
template <bool TRANSPOSE_EN>
inline void _llk_unpack_matmul_init_(
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const std::uint8_t ct_dim,
    const std::uint8_t rt_dim,
    const std::uint32_t kt_dim,
    const TensorShape& src_b_shape = DEFAULT_TENSOR_SHAPE,
    const TensorShape& src_a_shape = DEFAULT_TENSOR_SHAPE)
{
    static_assert((TRANSPOSE_EN == false), "TODO: Transpose srcA not available yet");
    LLK_ASSERT(validate_matmul_tensor_shapes_(src_b_shape, src_a_shape), "unsupported SrcB/SrcA TensorShape pair for matmul");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    if (src_b_shape.total_num_faces() == MAX_NUM_FACES && src_a_shape.total_num_faces() == MAX_NUM_FACES)
    {
        _llk_unpack_matmul_mop_config_(buf_desc_id_0, buf_desc_id_1, ct_dim, rt_dim, kt_dim);
    }
    else
    {
        _llk_unpack_matmul_face_mop_config_(buf_desc_id_0, buf_desc_id_1, ct_dim, rt_dim, kt_dim, src_b_shape, src_a_shape);
    }
}

/**
 * @brief Unpacks operands for matrix multiply: Input0 -> SrcB, Input1 -> SrcA.
 *
 * Unpacks for the rt and ct dims of input0 and input1 respectively, producing
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]. This unpacker only sets
 * up Input0 [rt_dim, 1] x Input1 [1, ct_dim]; kt_dim is assumed to be iterated over outside this call.
 * Constraints: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 *
 * @param ct_dim: Number of tiles in the column dimension for input1 of the matrix multiply.
 * @param rt_dim: Number of tiles in the row dimension for input0 of the matrix multiply.
 * @param kt_dim: Number of tiles in the common dimension between input0 and input1 of the matrix multiply.
 * @param start_l1_tile_idx_0/1: Start software-tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER1 -> SRCB, start_l1_tile_idx_1 -> UNPACKER0 -> SRCA.
 * @param src_b_shape: Input 0/SrcB tile shape. Default is a full 32x32 tile.
 * @param src_a_shape: Input 1/SrcA tile shape. Default is a full 32x32 tile.
 * @note Call @ref _llk_unpack_matmul_init_ with matching template args before this function.
 */
inline void _llk_unpack_matmul_(
    const std::uint8_t ct_dim,
    const std::uint8_t rt_dim,
    const std::uint32_t kt_dim,
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const TensorShape& src_b_shape = DEFAULT_TENSOR_SHAPE,
    const TensorShape& src_a_shape = DEFAULT_TENSOR_SHAPE)
{
    // Reset Dest counters for Unpacker to 0
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t src_b_tile_scale = _llk_unpack_matmul_src_tile_scale_(src_b_shape);
    const std::uint32_t src_a_tile_scale = _llk_unpack_matmul_src_tile_scale_(src_a_shape);

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t tile_idx_0 = start_l1_tile_idx_0 + (reuse_a ? (t * kt_dim) : 0);
        std::uint32_t tile_idx_1 = start_l1_tile_idx_1 + (reuse_a ? (0) : (t));

        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, tile_idx_0 * src_b_tile_scale);
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, tile_idx_1 * src_a_tile_scale);

        // Runs MOP
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
