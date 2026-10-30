// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cpack_common.h"
#include "llk_assert.h"
#include "llk_pack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Sets the L1 destination offset that carries untilize past each completed row of tiles.
 *
 * The Z counter walks tiles across a single row of tiles, but it does not advance between rows. To untilize more than
 * one row of tiles, program this register to account for the offset of every row already completed. The offset is
 * in units of 16 datums (one face row), so it is the number of face rows in a row of tiles.
 *
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_base_idx: Offset of the current row of tiles.
 * @note @ref _llk_pack_untilize_ calls this once per row of 32x32 tiles, before the MOP that writes that row.
 */
inline void _llk_pack_untilize_set_dst_offset_(const TensorShape& tensor_shape, const std::uint32_t l1_base_idx)
{
    const std::uint32_t dst_addr_offset_face_rows = l1_base_idx * tensor_shape.num_faces_c_dim;
    LLK_ASSERT(dst_addr_offset_face_rows <= THCON_PACKER0_REG0_UNTILIZE_DST_ADDR_OFFSET_MASK, "L1 offset exceeds the untilize dst address offset field");

    cfg_rmw(THCON_PACKER0_REG0_UNTILIZE_DST_ADDR_OFFSET_RMW, dst_addr_offset_face_rows);
}

/**
 * @brief Builds the MOP for pack untilize of contiguous tiles; works only with Packer 0.
 *
 * Packs out tile by tile while untilizing.
 *
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= MOP_LOOP_LEN_MAX).
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
 * @brief Builds the MOP for pack of a row of tiles, using strided reads to untilize.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= MOP_LOOP_LEN_MAX).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM>
inline void _llk_pack_untilize_strided_mop_config_(const std::uint8_t buf_desc_id, const TensorShape& tensor_shape)
{
    using ckernel::pack::PACR_STRIDE_ROWS;
    const std::uint32_t face_dest_rows      = padded_face_rows(tensor_shape);
    const std::uint32_t tile_dest_rows      = padded_tile_rows(tensor_shape);
    const std::uint32_t num_row_groups      = (tensor_shape.face_r_dim + PACR_STRIDE_ROWS - 1) / PACR_STRIDE_ROWS;
    const std::uint32_t l1_row_stride       = tensor_shape.num_faces_c_dim * FULL_CT_DIM;
    const std::uint32_t block_face_columns  = BLOCK_CT_DIM * tensor_shape.num_faces_c_dim;
    const std::uint32_t next_tile_dest_rows = tile_dest_rows - (tensor_shape.num_faces_c_dim - 1) * face_dest_rows;
    LLK_ASSERT(
        tensor_shape.num_faces_r_dim == 1 || (tensor_shape.num_faces_c_dim == 1 && num_row_groups * PACR_STRIDE_ROWS == face_dest_rows),
        "Strided pack untilize steps into the second face row only when it follows the first in Dest");
    const std::uint32_t num_outer_loops = tensor_shape.num_faces_r_dim * num_row_groups;

    // Packs one 4-row group of a face, then steps L1 one face column right and Dest to the next face along the output row.
    const auto pack_face_rows = [buf_desc_id](const std::uint32_t dest_rows_to_next_face)
    {
        return TT_OP_PACR_STRIDE(
            dest_rows_to_next_face / PACR_STRIDE_ROWS /*Src_Row_Idx_or_Inc_Mul4*/,
            1 /*Src_Row_Idx_Inc*/,
            0 /*L1_Tile_Idx_or_Tile_Idx_Inc, encodes +1*/,
            1 /*Tile_Idx_Inc*/,
            0 /*L1_16datums_Row_Index*/,
            buf_desc_id,
            0 /*PackerSel*/,
            0 /*ClrDatValid*/);
    };

    // With two face columns the second face of a tile is followed by the first face of the next tile.
    ckernel_template temp = (tensor_shape.num_faces_c_dim == 1)
                                ? ckernel_template(num_outer_loops, BLOCK_CT_DIM, pack_face_rows(next_tile_dest_rows))
                                : ckernel_template(num_outer_loops, BLOCK_CT_DIM, pack_face_rows(face_dest_rows), pack_face_rows(next_tile_dest_rows));
    temp.set_start_op(TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::ROW_SEL, p_pacr::PACK0, 0));
    temp.set_end_ops(
        TT_OP_INC_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, PACR_STRIDE_ROWS * l1_row_stride - block_face_columns),
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, PACR_STRIDE_ROWS));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Returns whether pack untilize uses strided pack (PACR_STRIDE): for every tile shape but 32x32.
 *
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @note When this returns true, program the buffer descriptor passed to @ref _llk_pack_untilize_init_ with L1AccessMode::Strided.
 */
constexpr bool _llk_pack_untilize_is_strided_(const TensorShape& tensor_shape)
{
    return tensor_shape.total_num_faces() != NUM_FACES;
}

/**
 * @brief Initializes the packer for pack untilize.
 *
 * Sets up the cfg registers with tile and tensor shape parameters which inform how TT_OP_PACR_UNTILIZE (32x32 tiles)
 * or TT_OP_PACR_STRIDE (every other tile shape) reads the data, then programs the MOP.
 *
 * @tparam FULL_CT_DIM: Number of tiles per block in c_dim * number of blocks = full c_dim size of the tensor.
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= MOP_LOOP_LEN_MAX); the block must also fit in the Dest space being packed.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @note Program buf_desc_id with L1AccessMode::Strided when @ref _llk_pack_untilize_is_strided_ is true.
 * @note The dst register is produced by the datacopy trio: on the unpack thread pair with @ref _llk_unpack_unary_operand_init_ (T0) and on the math thread with
 *       @ref _llk_math_eltwise_unary_datacopy_init_ (T1, A2D).
 * @note @ref _llk_pack_untilize_ is the matching execute call on this thread.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM>
inline void _llk_pack_untilize_init_(const std::uint8_t buf_desc_id, const TensorShape& tensor_shape)
{
    static_assert(FULL_CT_DIM % BLOCK_CT_DIM == 0, "FULL_CT_DIM must be divisible by BLOCK_CT_DIM");
    static_assert(BLOCK_CT_DIM <= MOP_LOOP_LEN_MAX, "BLOCK_CT_DIM exceeds the MOP inner loop length");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Unsupported tile shape for pack untilize");
    LLK_ASSERT(tensor_shape.total_num_faces() > 1 || tensor_shape.face_r_dim == FACE_R_DIM, "Pack untilize of 1-face tiles needs 16-row faces");

    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);

    if (!_llk_pack_untilize_is_strided_(tensor_shape))
    {
        ckernel::pack::pack_untilize_stride_cfg_u pk_cfg = {};

        pk_cfg.f.src_z_stride    = tensor_shape.total_num_faces() * tensor_shape.face_r_dim; // inc MATH DEST REG ptr by 64 16-datum rows
        pk_cfg.f.dst_z_stride    = tensor_shape.num_faces_c_dim;                             // inc L1 SRC ptr by 2 16-datum rows
        pk_cfg.f.stride_offset_0 = tensor_shape.num_faces_c_dim * FULL_CT_DIM;               // stride each row by 2*C_DIM 16-datum rows

        cfg[THCON_PACKER0_REG1_PACK_UNTILIZE_SRC_Z_STRIDE_ADDR32] = pk_cfg.val[0];
        cfg[THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_ADDR32]       = pk_cfg.val[1];

        _llk_pack_untilize_mop_config_<BLOCK_CT_DIM>(buf_desc_id);
        return;
    }

    ckernel::pack::pack_untilize_stride_cfg_u stride_cfg = {};
    stride_cfg.f.stride_offset_0                         = tensor_shape.num_faces_c_dim * FULL_CT_DIM; // one tensor row, in 16-datum units
    cfg[THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_ADDR32]  = stride_cfg.val[1];

    // Rows of a 4-row group past face_r_dim belong to the next face: mask them and skip their L1 writes.
    constexpr std::uint32_t row_group_mask = (1u << ckernel::pack::PACR_STRIDE_ROWS) - 1;
    const std::uint32_t row_mask           = (row_group_mask << tensor_shape.face_r_dim) & row_group_mask;
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_VAL_SOURCE_RMW, 0); // row k goes to start + k * STRIDE_OFFSET_0
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_COL_MASK_RMW, 0);
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_ROW_MASK_RMW, row_mask);
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_NO_WRITE_RMW, row_mask != 0);

    _llk_pack_untilize_strided_mop_config_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);
}

/**
 * @brief Packs out tiles and untilizes them; always uses Packer 0 for untilize.
 *
 * @param dest_idx: Index into the DEST register for a tile.
 * @param l1_tile_idx: Index into the L1 buffer for the first tile of the row, in tile widths.
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc.
 * @note Call @ref _llk_pack_untilize_init_ with the same tensor_shape before this function.
 */
inline void _llk_pack_untilize_(const std::uint32_t dest_idx, const std::uint32_t l1_tile_idx, const TensorShape& tensor_shape)
{
    if (!_llk_pack_untilize_is_strided_(tensor_shape))
    {
        // With semaphore math <-> pack sync and DstSync::Half, dest_register_offset flips at every Dest bank switch, so dest_idx
        // is offset by the number of tiles in one bank when bank 1 is current. With DstSync::Full, or with dvalid sync (which packs
        // relative to the current bank), dest_register_offset is always 0 and no offset is added.
        const std::uint32_t dest_bank1_offset_idx =
            (dest_register_offset == ckernel::trisc::DEST_REGISTER_HALF_SIZE) ? ckernel::DEST_NUM_TILES_FP16_HALF : ckernel::DEST_NUM_TILES_FP16_HALF >> 1;
        const std::uint32_t dest_reg_offset_idx = (dest_register_offset == 0) ? 0 : dest_bank1_offset_idx;
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, dest_reg_offset_idx + dest_idx);

        // The dst Z counter only walks tiles within the row; the L1 position goes through the dst address offset register,
        // because SET_DST_TILE_FACE_ROW_IDX under FACE_SEL passes only 11 bits to the counter.
        _llk_pack_untilize_set_dst_offset_(tensor_shape, l1_tile_idx);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        return;
    }

    const std::uint32_t l1_row_idx   = l1_tile_idx * tensor_shape.num_faces_c_dim;
    const std::uint32_t dest_row_idx = dest_idx * padded_tile_rows(tensor_shape);
    LLK_ASSERT(l1_row_idx < ckernel::trisc::TILE_SEL_IDX_LIMIT, "L1 row index exceeds the 18-bit SET_DST_TILE_FACE_ROW_IDX value field");
    LLK_ASSERT(dest_row_idx < ckernel::trisc::TILE_SEL_IDX_LIMIT, "Dest row index exceeds the 18-bit SET_SRC_TILE_FACE_ROW_IDX value field");
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_row_idx);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, dest_row_idx);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
