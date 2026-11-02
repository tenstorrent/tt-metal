// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Sets the L1 source offset that carries tilize past each completed row of tiles.
 *
 * The Z counter walks tiles across a single row of tiles, but it does not advance between rows. To tilize more than
 * one row of tiles, program this register to account for the offset of every row already completed. The offset is
 * in units of 16 datums (one face row), so it is the number of face rows in a row of tiles.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_base_idx: Offset of the current row of tiles.
 * @note Call this once per row of tiles, before the @ref _llk_unpack_tilize_ calls that tilize that row.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_tilize_set_src_offset_(const TensorShape& tensor_shape, const std::uint32_t l1_base_idx)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    const std::uint32_t src_addr_offset_face_rows = l1_base_idx * tensor_shape.num_faces_c_dim;

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TILIZE_SRC_ADDR_OFFSET_RMW, src_addr_offset_face_rows);
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TILIZE_SRC_ADDR_OFFSET_RMW, src_addr_offset_face_rows);
    }
}

/**
 * @brief Builds the MOP for unpack tilize of full 32x32 tiles using the fused HW instruction.
 *
 * Unpacks and tilizes block_ct_dim tiles per invocation into SrcA or SrcB; for DEST see llk_unpack_tilize_to_dest.h.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param block_ct_dim: Number of tiles unpacked per MOP invocation (MOP inner loop length).
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_tilize_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t block_ct_dim)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");
    LLK_ASSERT(block_ct_dim <= MOP_LOOP_LEN_MAX, "block_ct_dim exceeds the MOP loop length");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = block_ct_dim;

    const std::uint32_t unpack_tile_instrn = TT_OP_UNPACR_TILIZE(0, 0, 0 /*dst Z increment*/, 1 /*src Z increment*/, UNP_SEL, buf_desc_id, 1 /*SetDatValid*/);

    const std::uint32_t reset_src_reg_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 0 /*src Z increment*/, UNP_SEL, buf_desc_id, 1 /*SetDatValid*/);

    // FP32 datacopy goes through the math thread as ELWADD on SrcA+SrcB, which needs dvalid from both.
    if constexpr (EN_32BIT_DEST)
    {
        // FP32 datacopy uses ELWADD, which requires dvalid from both SrcA and SrcB
        // Set dvalid for the opposite unpacker (if using UNP_A, set dvalid for UNP_B and vice versa)
        constexpr std::uint32_t OPPOSITE_UNP                      = (UNP_SEL == p_unpacr::UNP_A) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
        constexpr static std::uint32_t set_opposite_dvalid_instrn = TT_OP_UNPACR_NOP(OPPOSITE_UNP, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*UNP_CLR_SRC*/);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, set_opposite_dvalid_instrn, unpack_tile_instrn);
        temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
        temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker to unpack-tilize a single operand by full 32x32 tiles.
 *
 * Packs the UNPACK_TILIZE stride fields into a single struct for a direct 32-bit cfg write, then
 * programs the MOP.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>. For DEST see
 *         @ref _llk_unpack_tilize_to_dest_init_.
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param full_ct_dim: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @param block_ct_dim: Number of tiles unpacked per MOP invocation (MOP inner loop length), a divisor of full_ct_dim.
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note On the math thread, pair with @ref _llk_math_eltwise_unary_datacopy_init_ (T1; tilize moves the tilized SrcA tile to dest); on the pack thread, pair
 * with
 *       @ref _llk_pack_init_ (T2). @ref _llk_unpack_tilize_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_tilize_init_(
    const std::uint32_t buf_desc_id, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    LLK_ASSERT(block_ct_dim >= 1 && full_ct_dim % block_ct_dim == 0, "block_ct_dim must divide full_ct_dim");

    // Pack all UNPACK_TILIZE stride fields into a single struct to perform a direct 32-bit cfg write
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride                       = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                  // Src (L1) counter increments in the UNPACR_TILIZE instruction
    unpk_cfg.f.dst_z_stride      = 1;                                             // col dim of a tile in dest reg (1 face)
    unpk_cfg.f.stride_val_source = 0;
    unpk_cfg.f.stride_offset_0   = full_ct_dim * tensor_shape.num_faces_c_dim; // how much to stride to go to next row within the same tile

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
        cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    else
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg[THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    _llk_unpack_tilize_mop_config_<UNP_SEL, EN_32BIT_DEST>(buf_desc_id, block_ct_dim);
}

/**
 * @brief Unpacks and tilizes block_ct_dim full 32x32 tiles, the MOP length set at init, from l1_tile_idx into SrcA or SrcB.
 *        For DEST see @ref _llk_unpack_tilize_to_dest_.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @param l1_tile_idx: Within-row column index of the first tile of the run. The L1 base is carried by
 *        @ref _llk_unpack_tilize_set_src_offset_, which the caller must set for the tile row.
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

    // The value field of the TT_SET_SRC_TILE_FACE_ROW_IDX instruction is 18 bits wide, but only values up to 11 bits are actually
    // passed on when using FACE_SEL; anything larger is truncated and aliases back to an earlier tile.
    LLK_ASSERT(l1_tile_idx < FACE_SEL_IDX_LIMIT, "l1_tile_idx exceeds the 11 bits the src face index can carry");

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, UNP_SEL, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, UNP_SEL, 0); // Clear face counter (the to-dest path leaves it non-zero)

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for strided unpack tilize; works for 32x32 and 16x32 tiles.
 *
 * Unpacks half a face with the strided instruction and increments the L1 counter.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor (row-major).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST, std::uint32_t FULL_CT_DIM>
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
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note @ref _llk_unpack_tilize_strided_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST, std::uint32_t FULL_CT_DIM>
inline void _llk_unpack_tilize_strided_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * tensor_shape.num_faces_c_dim);
    }
    else
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * tensor_shape.num_faces_c_dim);
    }
    _llk_unpack_tilize_strided_mop_config_<UNP_SEL, EN_32BIT_DEST, FULL_CT_DIM>(buf_desc_id, tensor_shape); // TODO: This throws a compile time error for UPK
                                                                                                            // but not PCK - why?
}

/**
 * @brief Unpacks and tilizes an entire tile using the stride instruction, for 32x32 or 16x32 tiles.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet.
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_tile_idx: c_dim index of the tile in L1.
 * @note Call @ref _llk_unpack_tilize_strided_init_ before this function to program the MOP.
 */
template <std::uint32_t UNP_SEL, std::uint32_t FULL_CT_DIM, bool EN_32BIT_DEST>
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
    if constexpr (UNP_SEL == p_unpacr::UNP_A && EN_32BIT_DEST)
    { // TODO pgardner: I don't think I'll need this for float32 dest
        TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B && EN_32BIT_DEST)
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
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
            stored in the buffer descriptor table, values = 0 - 16
 * @param block_ct_dim: Number of tiles in a row of a block.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_tilize_strided_mop_config_small_faces_(const std::uint32_t buf_desc_id, const std::uint32_t block_ct_dim)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert(!(EN_32BIT_DEST && UNP_SEL != p_unpacr::UNP_A), "If EN_32BIT_DEST then UNP_SEL should be UNP_A");
    LLK_ASSERT(block_ct_dim <= MOP_LOOP_LEN_MAX, "block_ct_dim exceeds the MOP loop length");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = block_ct_dim;
    constexpr std::uint32_t replay_buf_len = 2;

    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                TT_UNPACR0_STRIDE(
                    ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 0, 0, buf_desc_id, 0 /*Set Dvalid*/);
                TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 1, 0, 0, buf_desc_id, 1 /*Set Dvalid*/);
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                TT_UNPACR1_STRIDE(
                    ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 0, 0, buf_desc_id, 0 /*Set Dvalid*/);
                TT_UNPACR1_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 1, 0, 0, buf_desc_id, 1 /*Set Dvalid*/);
            });
    }

    if constexpr (EN_32BIT_DEST)
    {
        constexpr std::uint32_t OPPOSITE_UNP                      = (UNP_SEL == p_unpacr::UNP_B) ? p_unpacr::UNP_A : p_unpacr::UNP_B;
        constexpr static std::uint32_t set_opposite_dvalid_instrn = TT_OP_UNPACR_NOP(OPPOSITE_UNP, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*UNP_CLR_SRC*/);
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), set_opposite_dvalid_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker with stride values for strided tilize of small tiles (8x32, 4x32, 2x32, 1x32).
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param full_ct_dim: Number of tiles in a row of the input tensor (row-major).
 * @param block_ct_dim: Number of tiles in a row of a block, a divisor of full_ct_dim.
 * @note @ref _llk_unpack_tilize_strided_small_faces_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_tilize_strided_init_small_faces_(
    const std::uint32_t buf_desc_id, const TensorShape& tensor_shape, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim)
{
    LLK_ASSERT(block_ct_dim >= 1 && full_ct_dim % block_ct_dim == 0, "block_ct_dim must divide full_ct_dim");

    // Disable transpose
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, full_ct_dim * tensor_shape.num_faces_c_dim);
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, full_ct_dim * tensor_shape.num_faces_c_dim);
    }

    _llk_unpack_tilize_strided_mop_config_small_faces_<UNP_SEL, EN_32BIT_DEST>(buf_desc_id, block_ct_dim);
}

/**
 * @brief Unpacks and tilizes an entire tiny tile using the stride instruction; sizes 8x32, 4x32, 2x32, 1x32.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_tile_idx: Index into the L1 buffer for a tile.
 * @note Call @ref _llk_unpack_tilize_strided_init_small_faces_ before this function to program the MOP.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_tilize_strided_small_faces_(const TensorShape& tensor_shape, const std::uint32_t l1_tile_idx)
{
    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_tile_idx * tensor_shape.num_faces_c_dim);

    // Face 0 & 1
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
