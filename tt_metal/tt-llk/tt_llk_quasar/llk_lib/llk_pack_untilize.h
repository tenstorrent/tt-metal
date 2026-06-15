// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cpack_common.h"
#include "llk_pack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Builds the MOP for pack untilize of contiguous tiles; works only with Packer 0.
 *
 * Packs out tile by tile while untilizing.
 *
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= 8).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 */
template <std::uint32_t BLOCK_CT_DIM>
inline void _llk_pack_untilize_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = BLOCK_CT_DIM;

    std::uint32_t pack_instrn = TT_OP_PACR_UNTILIZE(0, 0, 1 /*inc Dst (L1) ctr*/, 1 /*inc Src ctr*/, 0 /*Packer 0 Sel*/, buf_desc_id, 0);
    // reset_src_and_dst_reg_instrn will reset Src (DEST reg) and Dst (L1) counters when the DEST reg bank is flipped aka at the last tile of the
    // block
    std::uint32_t reset_src_and_dst_reg_instrn =
        TT_OP_PACR_UNTILIZE(0, 3 /*Cntr_Reset_Mask*/, 0 /*inc Dst (L1) ctr*/, 0 /*inc Src ctr*/, 0, buf_desc_id, 0 /*Clr Dvalid*/);
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn);
    temp.set_last_outer_loop_instr(reset_src_and_dst_reg_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the packer for pack untilize of contiguous tiles.
 *
 * Sets up the cfg registers with tile and tensor shape parameters which inform how TT_OP_PACR_UNTILIZE
 * reads the data, then programs the MOP.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= 8).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @note The dst register is produced by the datacopy trio: on the unpack thread pair with @ref _llk_unpack_unary_operand_init_ (T0) and on the math thread with
 *       @ref _llk_math_eltwise_unary_datacopy_init_ (T1, A2D).
 * @note @ref _llk_pack_untilize_ is the matching execute call on this thread.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM>
inline void _llk_pack_untilize_init_(const std::uint8_t buf_desc_id, const TensorShape& tensor_shape)
{
    ckernel::pack::pack_untilize_stride_cfg_u pk_cfg = {};

    pk_cfg.f.src_z_stride    = tensor_shape.total_num_faces() * tensor_shape.face_r_dim; // inc MATH DEST REG ptr by 64 16-datum rows
    pk_cfg.f.dst_z_stride    = tensor_shape.num_faces_c_dim;                             // inc L1 SRC ptr by 2 16-datum rows
    pk_cfg.f.stride_offset_0 = tensor_shape.num_faces_c_dim * FULL_CT_DIM;               // stride each row by 2*C_DIM 16-datum rows

    cfg[THCON_PACKER0_REG1_PACK_UNTILIZE_SRC_Z_STRIDE_ADDR32] = pk_cfg.val[0];
    cfg[THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_ADDR32]       = pk_cfg.val[1];

    _llk_pack_untilize_mop_config_<BLOCK_CT_DIM>(buf_desc_id);
}

/**
 * @brief Packs out tiles and untilizes them; always uses Packer 0 for untilize.
 *
 * @param dest_idx: Index into the DEST register for a tile.
 * @param l1_tile_idx: Index into the L1 buffer for a tile.
 * @note Call @ref _llk_pack_untilize_init_ with matching template args before this function.
 */
inline void _llk_pack_untilize_(const std::uint32_t dest_idx, const std::uint32_t l1_tile_idx)
{
    // If we use semaphore math <-> pack synchronization, for pack untilize we need to pass the dest idx offset accounting for the current dest bank
    // dest_register_offset is updated at every dest bank switch when using semaphore math <-> pack synchronization and DstSync::Half, for DstSync::Full,
    // dest_register_offset is always 0 If we use dest dvalid math <-> pack synchronization, for pack untilize we need to pass the dest idx relative to the
    // current dest bank dest_register_offset is always 0 when using dest dvalid math <-> pack synchronization The number of tiles that fits in one dest bank is
    const std::uint32_t dest_bank1_offset_idx =
        (dest_register_offset == ckernel::trisc::DEST_REGISTER_HALF_SIZE) ? ckernel::DEST_NUM_TILES_FP16_HALF : ckernel::DEST_NUM_TILES_FP16_HALF >> 1;
    const std::uint32_t dest_reg_offset_idx = (dest_register_offset == 0) ? 0 : dest_bank1_offset_idx;
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, dest_reg_offset_idx + dest_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, l1_tile_idx);
    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for partial pack of a face, using strided reads to untilize.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam FACE_R_DIM: Number of rows per face.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t FACE_R_DIM>
inline void _llk_pack_untilize_strided_mop_config_(const std::uint8_t buf_desc_id, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t ROWS_READ      = 4;
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = FACE_R_DIM == 16 ? 3 : 1;

    std::uint32_t pack_instrn           = TT_OP_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    const std::uint32_t incr_L1_ptr =
        TT_OP_INC_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, ROWS_READ * tensor_shape.num_faces_c_dim * FULL_CT_DIM);
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, incr_L1_ptr);
    // temp.set_end_op(pack_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for partial pack of a face for 4x32 tiles, using strided reads to untilize.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam NUM_TILES_PER_BLOCK: Number of tiles processed per MOP run.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK>
inline void _llk_pack_untilize_strided_4x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK;

    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn                         = TT_OP_PACR_STRIDE(1, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR_STRIDE(0, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, pack_instrn);
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for partial pack of a face for 2x32 tiles, using strided reads to untilize.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam NUM_TILES_PER_BLOCK: Number of tiles processed per MOP run.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK>
inline void _llk_pack_untilize_strided_2x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK * 2;
    constexpr std::uint32_t replay_buf_len = 2;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id]
        {
            // Unpacks face 0 into dest offset 0
            TT_PACR_STRIDE(0, 0, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
            TTI_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 2);
        });

    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR_STRIDE(0, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for partial pack of a face for 1x32 tiles, using strided reads to untilize.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam NUM_TILES_PER_BLOCK: Number of tiles processed per MOP run.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK>
inline void _llk_pack_untilize_strided_1x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK;

    std::uint32_t pack_instrn                         = TT_OP_PACR0_ROW(0, 0, 0, 0, 1 /*inc L1 by 1 row*/, 1 /*inc DEST by 1 row*/, buf_desc_id, 0);
    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR0_ROW(0, 0, 0, 0, 1 /*inc L1 by 1 row*/, 0 /*inc DEST by 1 row*/, buf_desc_id, 0);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, pack_instrn);
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the packer for pack untilize of contiguous tiles using strided pack.
 *
 * Sets up the cfg registers with tile and tensor shape parameters which inform how TT_OP_PACR_STRIDE
 * reads the data, then dispatches to the per-FACE_R_DIM MOP builder.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam FACE_R_DIM: Number of rows per face, values = <1/2/4/8/16>
 * @tparam NUM_TILES_PER_BLOCK: Number of tiles processed per MOP run.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @note @ref _llk_pack_untilize_strided_ is the matching execute call on this thread.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t FACE_R_DIM, std::uint32_t NUM_TILES_PER_BLOCK>
inline void _llk_pack_untilize_strided_init_(const std::uint8_t buf_desc_id, const TensorShape& tensor_shape)
{
    static_assert(
        FACE_R_DIM == 1 || FACE_R_DIM == 2 || FACE_R_DIM == 4 || FACE_R_DIM == 8 || FACE_R_DIM == 16,
        "FACE_R_DIM must be 1, 2, 4, 8, or 16 for strided pack untilize");
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_VAL_SOURCE_RMW, 0); // sel STRIDE_OFFSET_0
    if constexpr (FACE_R_DIM != 1)
    {
        ckernel::pack::pack_untilize_stride_cfg_u stride_cfg = {};
        stride_cfg.f.stride_offset_0 = tensor_shape.num_faces_c_dim * FULL_CT_DIM; // stride each row by 2*FULL_CT_DIM 16-datum rows
        cfg[THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_ADDR32] = stride_cfg.val[1];
    }

    if constexpr (FACE_R_DIM >= 8) // 32x32, 16x32, 8x32
    {
        _llk_pack_untilize_strided_mop_config_<FULL_CT_DIM, FACE_R_DIM>(buf_desc_id, tensor_shape);
    }
    else if constexpr (FACE_R_DIM == 4) // 4x32
    {
        _llk_pack_untilize_strided_4x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK>(buf_desc_id);
    }
    else if constexpr (FACE_R_DIM == 2) // 2x32
    {
        cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_NO_WRITE_RMW, 1); // mask off last 2 rows
        cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_ROW_MASK_RMW, 0x1100);
        _llk_pack_untilize_strided_2x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK>(buf_desc_id);
    }
    else // 1x32
    {
        _llk_pack_untilize_strided_1x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK>(buf_desc_id);
    }
}

/**
 * @brief Packs a single tile, untilizing it using strided pack.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @param l1_tile_idx: Index into the L1 output buffer for the tile.
 * @param src_tile_idx: Tile index into the source (math destination) register.
 * @note Call @ref _llk_pack_untilize_strided_init_ with matching template args before this function.
 */
template <std::uint32_t FULL_CT_DIM>
inline void _llk_pack_untilize_strided_(
    const std::uint8_t buf_desc_id, const TensorShape& tensor_shape, const std::uint32_t l1_tile_idx, const std::uint32_t src_tile_idx)
{
    const std::uint32_t f1_row_idx = (tensor_shape.num_faces_c_dim == 1)
                                         ? l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM
                                         : l1_tile_idx * tensor_shape.num_faces_c_dim + 1;

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, src_tile_idx * tensor_shape.face_r_dim * tensor_shape.total_num_faces());
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_tile_idx * tensor_shape.num_faces_c_dim);

    // Face 0
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    // Face 1
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, f1_row_idx);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    if (tensor_shape.total_num_faces() == ckernel::trisc::NUM_FACES)
    {
        TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
        // Face 2
        TT_SET_DST_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL,
            p_pacr::PACK0,
            l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

        // Face 3
        TT_SET_DST_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL,
            p_pacr::PACK0,
            l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM + 1);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        TT_PACR_STRIDE(0, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    }
    else
    {
        TT_PACR_STRIDE(0, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    }
}

/**
 * @brief Runs one strided pack untilize MOP invocation for 4x32 tiles.
 *
 * @note Call @ref _llk_pack_untilize_strided_init_ (with FACE_R_DIM = 4) with matching template args before this function.
 */
inline void _llk_pack_untilize_strided_4x32_()
{
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
