// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Builds the MOP for unpack tilize of full 32x32 tiles using the fused HW instruction.
 *
 * Unpacks and tilizes block_ct_dim tiles per invocation; works for SrcA/B and DEST.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param block_ct_dim: Number of tiles unpacked per MOP invocation (MOP inner loop length).
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_tilize_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t block_ct_dim)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = block_ct_dim;

    // For UNP_DEST, don't set dvalid on individual tiles, the section_done signal handles it.
    // Setting dvalid per tile would cause the packer to start (and ZEROACC) before all tiles are in DEST.
    constexpr std::uint32_t SET_DVALID = (UNP_SEL == p_unpacr::UNP_DEST) ? 0 : 1;
    std::uint32_t unpack_tile_instrn   = TT_OP_UNPACR_TILIZE(0, 0, 0 /*dst Z increment*/, 1 /*src Z increment*/, UNP_SEL, buf_desc_id, SET_DVALID);

    std::uint32_t reset_src_reg_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 0 /*src Z increment*/, UNP_SEL, buf_desc_id, SET_DVALID);

    // This path is exclusively for FP32 datacopy via math thread (ELWADD on SrcA+SrcB),
    // not for UNP_DEST where data goes directly to DEST without involving the math thread.
    if constexpr (IS_32b_DEST_EN && (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_B))
    {
        // FP32 datacopy uses ELWADD, which requires dvalid from both SrcA and SrcB
        // Set dvalid for the opposite unpacker (if using UNP_A, set dvalid for UNP_B and vice versa)
        constexpr std::uint32_t OPPOSITE_UNP                      = (UNP_SEL == p_unpacr::UNP_A) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
        constexpr static std::uint32_t set_opposite_dvalid_instrn = TT_OP_UNPACR_NOP(OPPOSITE_UNP, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*UNP_CLR_SRC*/);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, set_opposite_dvalid_instrn, unpack_tile_instrn);
        // When inner loop len == 1, last_outer replaces the only loop op — skip it so the
        // real tilize (+ opposite dvalid) still runs. Counters are re-set in `_llk_unpack_tilize_`.
        if (block_ct_dim > 1)
        {
            temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        }
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
        // Same: with block_ct_dim==1, last_outer would replace the sole UNPACR_TILIZE.
        if (block_ct_dim > 1)
        {
            temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        }
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker to unpack-tilize a single operand by full 32x32 tiles.
 *
 * Packs the UNPACK_TILIZE stride fields into a single struct for a direct 32-bit cfg write, then
 * programs the MOP.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param full_ct_dim: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @param block_ct_dim: Number of tiles unpacked per MOP invocation (MOP inner loop length).
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note On the math thread, pair with @ref _llk_math_eltwise_unary_datacopy_init_ (T1; tilize moves the tilized SrcA tile to dest); on the pack thread, pair with
 *       @ref _llk_pack_init_ (T2). @ref _llk_unpack_tilize_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_tilize_init_(
    const std::uint32_t buf_desc_id, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    // Pack all UNPACK_TILIZE stride fields into a single struct to perform a direct 32-bit cfg write
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride      = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                 // Src (L1) counter increments in the UNPACR_TILIZE instruction
    unpk_cfg.f.dst_z_stride      = 1;           // col dim of a tile in dest reg (1 face)
    unpk_cfg.f.stride_val_source = 0;
    unpk_cfg.f.stride_offset_0   = full_ct_dim * tensor_shape.num_faces_c_dim; // how much to stride to go to next row within the same tile

    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg[THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    _llk_unpack_tilize_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, block_ct_dim);
}

/**
 * @brief Unpacks and tilizes a single full 32x32 tile; works for UNP_A, UNP_B, UNP_DEST.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @param l1_tile_idx: Index into the L1 buffer for a tile.
 * @note Call @ref _llk_unpack_tilize_init_ with matching template args before this function.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_tilize_(const std::uint32_t l1_tile_idx)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    // UNP_DEST shares UNP_A's hardware path, so use UNP_A for counter instructions
    constexpr std::uint32_t CNT_SEL = (UNP_SEL == p_unpacr::UNP_DEST) ? p_unpacr::UNP_A : UNP_SEL;
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, CNT_SEL, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, CNT_SEL, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, CNT_SEL, 0); // Clear face counter (block path leaves it non-zero)

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for batched tilize directly into the DEST register.
 *
 * Sets up a single-instruction MOP: UNPACR_TILIZE with Dst_Z_Cntr_inc=1. UNPACR_TILIZE for UNP_DEST
 * uses its internal Z counter for DEST addressing. DST_Z_STRIDE must be set to NUM_FACES so each Z
 * increment advances by one full tile.
 *
 * @tparam BLOCK_CT_DIM: Number of tiles per row to process in one MOP invocation.
 * @param buf_desc_id: The buffer descriptor ID, values = 0 - 16
 */
template <std::uint32_t BLOCK_CT_DIM>
inline void _llk_unpack_tilize_block_mop_config_(const std::uint32_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = BLOCK_CT_DIM;

    // Tilize one tile to DEST, then advance DEST Z counter by DST_Z_STRIDE (= total_num_faces).
    // No dvalid, the section_done signal handles synchronization after the full block.
    // Src Z auto-increments by SRC_Z_STRIDE (= num_faces_c_dim) to advance to the next tile in L1.
    std::uint32_t unpack_tile_instrn = TT_OP_UNPACR_TILIZE(0, 0, 1 /*dst Z inc*/, 1 /*src Z inc*/, p_unpacr::UNP_DEST, buf_desc_id, 0 /*no dvalid*/);

    // Last tile: reset counters, no src/dst Z increment (counters will be set explicitly for next row).
    std::uint32_t reset_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z inc*/, 0 /*src Z inc*/, p_unpacr::UNP_DEST, buf_desc_id, 0 /*no dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
    temp.set_last_outer_loop_instr(reset_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for batched tilize directly into the DEST register.
 *
 * Configures UNP_A (shared with UNP_DEST) stride registers and programs the block MOP. DST_Z_STRIDE
 * is set to NUM_FACES so each UNPACR_TILIZE Z increment advances by one full tile in DEST (instead of
 * 1 face as in the single-tile path).
 *
 * @tparam FULL_CT_DIM: Number of tiles in a full row of the input tensor.
 * @tparam BLOCK_CT_DIM: Number of tiles per row to process in one MOP invocation.
 * @param buf_desc_id: The buffer descriptor ID, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note Caller must ensure BLOCK_RT_DIM * BLOCK_CT_DIM <= dest_size_in_tiles, since all tiles
 *       in the block are accumulated in DEST across rows before a single section_done is issued.
 * @note @ref _llk_unpack_tilize_block_ is the matching execute call on this thread.
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM>
inline void _llk_unpack_tilize_block_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride      = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face)
    // Z stride unit = face_r_dim datums. Each tile = total_num_faces faces × face_r_dim rows per face.
    unpk_cfg.f.dst_z_stride      = tensor_shape.total_num_faces() * tensor_shape.face_r_dim; // stride between tiles in DEST
    unpk_cfg.f.stride_val_source = 0;
    unpk_cfg.f.stride_offset_0   = FULL_CT_DIM * tensor_shape.num_faces_c_dim; // stride to next row within same tile

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
    cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
    cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];

    _llk_unpack_tilize_block_mop_config_<BLOCK_CT_DIM>(buf_desc_id);
}

/**
 * @brief Executes batched tilize of one row of tiles directly into the DEST register.
 *
 * Sets the L1 source and DEST counters once, then runs the MOP which processes BLOCK_CT_DIM tiles.
 * The MOP auto-advances both source (via Src_Z_Cntr_inc) and dest (via Dst_Z_Cntr_inc with
 * DST_Z_STRIDE=NUM_FACES). Call once per tile row, then call @ref _llk_unpack_dest_dvalid_section_done_
 * after all rows.
 *
 * @param l1_face_idx: Face-level index into the L1 buffer for the start of this tile row.
 * @param dest_tile_idx: Tile index within DEST for the first tile of this row.
 * @note Call @ref _llk_unpack_tilize_block_init_ before this function to program the MOP.
 */
inline void _llk_unpack_tilize_block_(const std::uint32_t l1_face_idx, const std::uint32_t dest_tile_idx)
{
    // UNP_DEST shares UNP_A's counter hardware.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_unpacr::UNP_A, l1_face_idx);
    // UNPACR_TILIZE Z counter controls DEST positioning; DST_Z_STRIDE=total_num_faces so each
    // Z unit = one tile.  Reset face counter to dest_tile_idx so the first tile in this row
    // lands at the correct DEST slot.
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_unpacr::UNP_A, dest_tile_idx);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for strided unpack tilize; works for 32x32 and 16x32 tiles.
 *
 * Unpacks half a face with the strided instruction and increments the L1 counter.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor (row-major).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN, std::uint32_t FULL_CT_DIM>
inline void _llk_unpack_tilize_strided_mop_config_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    std::uint32_t unpack_half_face_instrn =
        (UNP_SEL == p_unpacr::UNP_A)
            ? TT_OP_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id, 0 /*Set Dvalid*/)
            : TT_OP_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id, 0 /*Set Dvalid*/);
    const std::uint32_t increment_half_face_instrn =
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, tensor_shape.num_faces_c_dim * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_half_face_instrn, increment_half_face_instrn);
    temp.set_end_op(unpack_half_face_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker with stride values for strided tilize of 32x32 or 16x32 tiles.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note @ref _llk_unpack_tilize_strided_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN, std::uint32_t FULL_CT_DIM>
inline void _llk_unpack_tilize_strided_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * tensor_shape.num_faces_c_dim);
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * tensor_shape.num_faces_c_dim);
    }
    _llk_unpack_tilize_strided_mop_config_<UNP_SEL, IS_32b_DEST_EN, FULL_CT_DIM>(buf_desc_id, tensor_shape); // TODO: This throws a compile time error for UPK
                                                                                                             // but not PCK - why?
}

/**
 * @brief Unpacks and tilizes an entire tile using the stride instruction, for 32x32 or 16x32 tiles.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_tile_idx: c_dim index of the tile in L1.
 * @note Call @ref _llk_unpack_tilize_strided_init_ before this function to program the MOP.
 */
template <std::uint32_t UNP_SEL, std::uint32_t FULL_CT_DIM, bool IS_32b_DEST_EN>
inline void _llk_unpack_tilize_strided_(const TensorShape& tensor_shape, const std::uint32_t l1_tile_idx)
{
    const std::uint32_t f1_row_idx =
        (tensor_shape.num_faces_c_dim == 1 ? l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM
                                           : l1_tile_idx * tensor_shape.num_faces_c_dim + 1);

    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_tile_idx * tensor_shape.num_faces_c_dim);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);

    // Face 0
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, f1_row_idx); // Set L1 ptr

    if (tensor_shape.total_num_faces() == ckernel::trisc::NUM_FACES)
    {
        // Face 1
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer); // Unpack half face and increment L1 ptr

        // Face 2
        TT_SET_SRC_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL,
            UNP_SEL,
            l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

        // Face 3
        TT_SET_SRC_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL,
            UNP_SEL,
            l1_tile_idx * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * FULL_CT_DIM + 1);
    }
    if constexpr (UNP_SEL == p_unpacr::UNP_A && IS_32b_DEST_EN)
    { // TODO pgardner: I don't think I'll need this for float32 dest
        TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B && IS_32b_DEST_EN)
    {
        TTI_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/);
    }

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    { // This is ugly but will likely change soon anyways
        TTI_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 0 /*Set Dvalid*/);
        TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, tensor_shape.num_faces_c_dim * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);
        TTI_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 1 /*Set Dvalid*/);
    }
    else
    {
        TTI_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 0 /*Set Dvalid*/);
        TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, tensor_shape.num_faces_c_dim * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);
        TTI_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 1 /*Set Dvalid*/);
    }
}

/**
 * @brief Builds the MOP for strided unpack tilize of small tiles (8x32, 4x32, 2x32, 1x32).
 *
 * Unpacks half a face with the strided instruction and increments the L1 counter.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor (row-major).
 * @tparam ROWS_READ: Number of rows read by one UNPACR0_STRIDE call.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN, std::uint32_t FULL_CT_DIM, std::uint32_t ROWS_READ>
inline void _llk_unpack_tilize_strided_mop_config_small_faces_(const std::uint32_t buf_desc_id)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert(!(IS_32b_DEST_EN && UNP_SEL != p_unpacr::UNP_A), "If IS_32b_DEST_EN then UNP_SEL should be UNP_A");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = FULL_CT_DIM;

    std::uint32_t unpack_face0_instrn =
        TT_OP_UNPACR0_STRIDE(ROWS_READ /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 0, 0, buf_desc_id, 0 /*Set Dvalid*/);
    std::uint32_t unpack_face1_instrn = TT_OP_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 1, 0, 0, buf_desc_id, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_face0_instrn, unpack_face1_instrn);

    // FP32 datacopy uses ELWADD, which requires datavalid from both SrcA and SrcB, so need to add SrcB datavalid
    if constexpr (UNP_SEL == p_unpacr::UNP_A && IS_32b_DEST_EN)
    {
        temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B && IS_32b_DEST_EN)
    {
        temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker with stride values for strided tilize of small tiles (8x32, 4x32, 2x32, 1x32).
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @tparam ROWS_READ: Number of rows read by one UNPACR0_STRIDE call.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note @ref _llk_unpack_tilize_strided_small_faces_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN, std::uint32_t FULL_CT_DIM, std::uint32_t ROWS_READ>
inline void _llk_unpack_tilize_strided_init_small_faces_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
    cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
    cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * tensor_shape.num_faces_c_dim);
    _llk_unpack_tilize_strided_mop_config_small_faces_<UNP_SEL, IS_32b_DEST_EN, FULL_CT_DIM, ROWS_READ>(buf_desc_id);
}

/**
 * @brief Unpacks and tilizes an entire tiny tile using the stride instruction; sizes 8x32, 4x32, 2x32, 1x32.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note Call @ref _llk_unpack_tilize_strided_init_small_faces_ before this function to program the MOP.
 */
template <std::uint32_t UNP_SEL, std::uint32_t FULL_CT_DIM>
inline void _llk_unpack_tilize_strided_small_faces_(const TensorShape& tensor_shape)
{
    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset

    // Face 0 & 1
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
