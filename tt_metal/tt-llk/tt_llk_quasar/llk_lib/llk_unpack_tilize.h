// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Sets the L1 source offset that UNPACR_TILIZE reads from.
 *
 * The Z counter walks tiles across a single row of tiles, but it does not advance between rows, so the L1 position of
 * the first tile of a run goes through this register. The offset is in units of 16 datums (one face row), so it is
 * l1_base_idx times the number of face columns. Drains the unpacker first, since the previous run may still be reading
 * this CFG word.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @param l1_base_idx: Offset of the first tile of the run, in tile-width rows of the L1 buffer.
 * @note @ref _llk_unpack_tilize_ calls this for 32x32 tiles, before the MOP that tilizes the run.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_tilize_set_src_offset_(const TensorShape& tensor_shape, const std::uint32_t l1_base_idx)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    const std::uint32_t src_addr_offset_face_rows = l1_base_idx * tensor_shape.num_faces_c_dim;

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        LLK_ASSERT(
            src_addr_offset_face_rows <= (THCON_UNPACKER0_REG0_TILIZE_SRC_ADDR_OFFSET_MASK >> THCON_UNPACKER0_REG0_TILIZE_SRC_ADDR_OFFSET_SHAMT),
            "L1 offset exceeds the tilize src address offset field");
        _llk_stall_cfg_on_<p_stall::UNPACK0>();
        cfg_rmw(THCON_UNPACKER0_REG0_TILIZE_SRC_ADDR_OFFSET_RMW, src_addr_offset_face_rows);
    }
    else
    {
        LLK_ASSERT(
            src_addr_offset_face_rows <= (THCON_UNPACKER1_REG0_TILIZE_SRC_ADDR_OFFSET_MASK >> THCON_UNPACKER1_REG0_TILIZE_SRC_ADDR_OFFSET_SHAMT),
            "L1 offset exceeds the tilize src address offset field");
        _llk_stall_cfg_on_<p_stall::UNPACK1>();
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
 * @brief Builds the MOP for strided unpack tilize of a run of tiles: for every tile shape but 32x32.
 *
 * A tile is read as one or two face columns (32 datums wide tiles have two), each face column top to bottom in groups of
 * UNPACR_STRIDE_MAX_ROWS L1 rows, one UNPACR_STRIDE per group. Every group lands UNPACR_STRIDE_MAX_ROWS SrcA rows below
 * the previous one, so a face shorter than that is padded to UNPACR_STRIDE_MAX_ROWS rows (see tiny_face_stride), the
 * layout the math datacopy reads. The last group of a tile sets dvalid and resets the SrcA row counter. The per-tile
 * sequence goes into the replay buffer and the MOP replays it once per tile.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam EN_32BIT_DEST: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param full_ct_dim: Number of tiles in a row of the input tensor (row-major).
 * @param block_ct_dim: Number of tiles unpacked per MOP invocation (MOP inner loop length).
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 */
template <std::uint32_t UNP_SEL, bool EN_32BIT_DEST>
inline void _llk_unpack_tilize_strided_mop_config_(
    const std::uint32_t buf_desc_id, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");
    LLK_ASSERT(!EN_32BIT_DEST || UNP_SEL == p_unpacr::UNP_A, "Strided unpack tilize with EN_32BIT_DEST supports only UNP_A");
    LLK_ASSERT(block_ct_dim <= MOP_LOOP_LEN_MAX, "block_ct_dim exceeds the MOP loop length");

    constexpr std::uint32_t GROUP_ROWS     = ckernel::unpack::UNPACR_STRIDE_MAX_ROWS;
    const std::uint32_t num_faces_c_dim    = tensor_shape.num_faces_c_dim;
    const std::uint32_t face_column_rows   = tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;
    const std::uint32_t num_row_groups     = (face_column_rows + GROUP_ROWS - 1) / GROUP_ROWS;
    const std::uint32_t group_l1_rows      = GROUP_ROWS * num_faces_c_dim * full_ct_dim;
    const std::uint32_t face_column_rewind = (0u - (num_row_groups - 1) * group_l1_rows) & (ckernel::trisc::TILE_SEL_IDX_LIMIT - 1);
    LLK_ASSERT(
        tensor_shape.num_faces_r_dim == 1 || (num_faces_c_dim == 1 && tensor_shape.face_r_dim % GROUP_ROWS == 0),
        "Strided unpack tilize reads the second face row in the same face column only when it follows the first in SrcA");

    // Per face column: every group but the last steps L1 one group down with an INC; the last group steps L1 one
    // face column right through its own index increment, and an INC rewinds the groups it went down.
    const std::uint32_t replay_buf_len = num_faces_c_dim * (num_row_groups == 1 ? 1 : 2 * num_row_groups);

    // Unpacks one group of rows into SrcA/SrcB and moves the SrcA/SrcB row counter by src_y_inc (0 resets it).
    const auto unpack_group = [buf_desc_id](const std::uint32_t src_y_inc, const bool next_face_column, const bool set_dvalid)
    {
        if constexpr (UNP_SEL == p_unpacr::UNP_A)
        {
            return TT_OP_UNPACR0_STRIDE(
                src_y_inc,
                0 /*L1_Tile_Idx_or_Tile_Idx_Inc, encodes +1*/,
                next_face_column,
                0 /*Row_Mask_Reg_Sel*/,
                0 /*L1_16datums_Row_Index*/,
                buf_desc_id,
                set_dvalid);
        }
        else
        {
            return TT_OP_UNPACR1_STRIDE(
                src_y_inc,
                0 /*L1_Tile_Idx_or_Tile_Idx_Inc, encodes +1*/,
                next_face_column,
                0 /*Row_Mask_Reg_Sel*/,
                0 /*L1_16datums_Row_Index*/,
                buf_desc_id,
                set_dvalid);
        }
    };

    load_replay_buf(
        0 /*start*/,
        replay_buf_len,
        false /*exec_while_loading*/,
        0 /*set_mutex*/,
        0 /*last*/,
        [=]
        {
            for (std::uint32_t face_column = 0; face_column < num_faces_c_dim; face_column++)
            {
                for (std::uint32_t group = 0; group < num_row_groups; group++)
                {
                    const bool last_group   = (group == num_row_groups - 1);
                    const bool last_of_tile = last_group && (face_column == num_faces_c_dim - 1);
                    TT_INSN(unpack_group(last_of_tile ? 0 : GROUP_ROWS, last_group, last_of_tile));
                    if (num_row_groups > 1)
                    {
                        TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, last_group ? face_column_rewind : group_l1_rows);
                    }
                }
            }
        });

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = block_ct_dim;
    const std::uint32_t unpack_tile_instrn = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);

    if constexpr (EN_32BIT_DEST)
    {
        constexpr std::uint32_t OPPOSITE_UNP                      = (UNP_SEL == p_unpacr::UNP_B) ? p_unpacr::UNP_A : p_unpacr::UNP_B;
        constexpr static std::uint32_t set_opposite_dvalid_instrn = TT_OP_UNPACR_NOP(OPPOSITE_UNP, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*UNP_CLR_SRC*/);
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn, set_opposite_dvalid_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Returns whether unpack tilize uses strided unpack (UNPACR_STRIDE): for every tile shape but 32x32.
 *
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note When this returns true, program the buffer descriptor passed to @ref _llk_unpack_tilize_init_ with L1AccessMode::Strided.
 */
constexpr bool _llk_unpack_tilize_is_strided_(const TensorShape& tensor_shape)
{
    return tensor_shape.total_num_faces() != NUM_FACES;
}

/**
 * @brief Initializes the unpacker to unpack-tilize a single operand into SrcA or SrcB.
 *
 * Packs the UNPACK_TILIZE stride fields into a single struct for a direct 32-bit cfg write, then programs the MOP:
 * UNPACR_TILIZE for 32x32 tiles, UNPACR_STRIDE for every other tile shape.
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
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");
    LLK_ASSERT(block_ct_dim >= 1 && full_ct_dim % block_ct_dim == 0, "block_ct_dim must divide full_ct_dim");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Unsupported tile shape for unpack tilize");
    LLK_ASSERT(tensor_shape.total_num_faces() > 1 || tensor_shape.face_r_dim == FACE_R_DIM, "Unpack tilize of 1-face tiles needs 16-row faces");

    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, UNP_SEL, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, UNP_SEL, 0);

    // Pack all UNPACK_TILIZE stride fields into a single struct to perform a direct 32-bit cfg write. The write also
    // clears STRIDE_NO_WRITE, which a strided reduce may have left set.
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride                       = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                  // Src (L1) counter increments in the UNPACR_TILIZE instruction
    unpk_cfg.f.dst_z_stride      = 1;                                             // col dim of a tile in dest reg (1 face)
    unpk_cfg.f.stride_val_source = 0;                                             // row k goes to start + k * STRIDE_OFFSET_0
    unpk_cfg.f.stride_offset_0   = full_ct_dim * tensor_shape.num_faces_c_dim;    // how much to stride to go to next row within the same tile

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

    if (!_llk_unpack_tilize_is_strided_(tensor_shape))
    {
        _llk_unpack_tilize_mop_config_<UNP_SEL, EN_32BIT_DEST>(buf_desc_id, block_ct_dim);
        return;
    }

    // The strided MOP selects row mask 0 and relies on it and the column mask staying at their reset value of 0 (keep
    // every row and column of a group); no LLK writes either.
    _llk_unpack_tilize_strided_mop_config_<UNP_SEL, EN_32BIT_DEST>(buf_desc_id, full_ct_dim, block_ct_dim, tensor_shape);
}

/**
 * @brief Unpacks and tilizes block_ct_dim tiles, the MOP length set at init, from l1_tile_idx into SrcA or SrcB.
 *        For DEST see @ref _llk_unpack_tilize_to_dest_.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @param l1_tile_idx: Index into the L1 buffer for the first tile of the run, in tile-width rows: the offset of its row
 *        of tiles plus its column index within that row.
 * @param tensor_shape: Tile shape info: num faces, face row/col dim, etc.
 * @note Call @ref _llk_unpack_tilize_init_ with matching template args and the same tensor_shape before this function.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_tilize_(const std::uint32_t l1_tile_idx, const TensorShape& tensor_shape)
{
    if (!_llk_unpack_tilize_is_strided_(tensor_shape))
    {
        // The src Z counter only walks tiles within the run; the L1 position goes through the src address offset
        // register, because SET_SRC_TILE_FACE_ROW_IDX under FACE_SEL passes only 11 bits to the counter.
        _llk_unpack_tilize_set_src_offset_<UNP_SEL>(tensor_shape, l1_tile_idx);
        TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, UNP_SEL, 0);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, UNP_SEL, 0); // Clear face counter (the to-dest path leaves it non-zero)
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        return;
    }

    // Strided buffer descriptors index L1 by 16-datum rows.
    const std::uint32_t l1_row_idx = l1_tile_idx * tensor_shape.num_faces_c_dim;
    LLK_ASSERT(l1_row_idx < ckernel::trisc::TILE_SEL_IDX_LIMIT, "L1 row index exceeds the 18-bit SET_SRC_TILE_FACE_ROW_IDX value field");
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_row_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
