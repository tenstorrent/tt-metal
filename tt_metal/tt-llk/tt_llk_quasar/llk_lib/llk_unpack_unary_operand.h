// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Returns the NOP Tensix instruction for unused unpacker engine
 *
 * @tparam UNP_SEL: unpacker engine in use
 */
template <std::uint32_t UNP_SEL>
constexpr std::uint32_t nop_insn_for_unused_unpacker_engine()
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL must be UNP_A or UNP_B");
    constexpr auto unpacr_engine = UNP_SEL == p_unpacr::UNP_A ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    return TT_OP_UNPACR_NOP(unpacr_engine, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, 0 /*Src_ClrVal_Ctrl*/, 0 /*Nop_type*/);
}

/**
 * @brief MOP configuration for unpack of unary operations
 * @details Sets up MOP for unpacking a single operand by tiles
 * Specialized for tiny-tile unpack where each face is a separate tile in HW.
 * Works for any unpack resource
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32-bit mode
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for a single operand
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_variable_tile_size_mop_config_(
    const std::uint32_t buf_desc_id, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    // TODO: Implement Unpack to dest for tiny tiles
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    const std::uint32_t MOP_OUTER_LOOP = num_tiles;
    const std::uint32_t MOP_INNER_LOOP = tensor_shape.total_num_faces();

    // For UNP_A/UNP_B: Dst Tile Idx Inc = 0 so each face overwrites the same SrcA/B tile slot.
    // Dvalid is set per face so math can consume each face before the next one arrives.
    // For UNP_DEST: Dst Tile Idx Inc = 1 to place faces at consecutive dest positions (no math involved).
    std::uint32_t unpack_tile_instrn;
    std::uint32_t unpack_tile_w_dvalid_instrn;
    std::uint32_t reset_dest_tile_cnt_instrn = TT_OP_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0 /*Value*/);

    std::uint32_t dest_tile_idx_inc = (static_cast<std::uint32_t>(tensor_shape.face_r_dim) < (FACE_R_DIM >> 1))
                                          ? (FACE_R_DIM >> (rows_log2(static_cast<std::uint32_t>(tensor_shape.face_r_dim)) + 1))
                                          : 1;

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        unpack_tile_instrn          = TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
        unpack_tile_w_dvalid_instrn = TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        unpack_tile_instrn          = TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
        unpack_tile_w_dvalid_instrn = TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);

    // TODO: Figure out why setting unpack_tile_w_dvalid_instrn using set_last_outer_loop_instr did not work
    temp.set_inner_loop_len(MOP_INNER_LOOP - 1); // Inner loop iterates over num_faces-1 where the dvalid unpacking is done as the END_OP
    temp.set_end_ops(unpack_tile_w_dvalid_instrn, reset_dest_tile_cnt_instrn);

    // If IS_32b_DEST_EN and UNP_SEL = UNP_A, zero out the SRCB reg
    // The only test in which there is a unary upk to SRCA with 32b DF is the datacopy kernel, which uses ELWADD
    if constexpr (IS_32b_DEST_EN)
    {
        const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
        temp.set_start_op(clr_unused_unpacr_engine);
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for unpack of unary operations
 * @details Sets up MOP for unpacking a single operand by tiles
 * works for any unpack resource
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32-bit mode
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for a single operand
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");

    const std::uint32_t MOP_OUTER_LOOP     = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_UNPACR_FACE_INC
    std::uint32_t unpack_tile_instrn;
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        unpack_tile_instrn = TT_OP_UNPACR0_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        unpack_tile_instrn = TT_OP_UNPACR1_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_DEST)
    {
        unpack_tile_instrn = TT_OP_UNPACR_DEST_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);

    // If IS_32b_DEST_EN and UNP_SEL = UNP_A, zero out the SRCB reg
    // The only test in which there is a unary upk to SRCA with 32b DF is the datacopy kernel, which uses ELWADD
    if constexpr (IS_32b_DEST_EN && (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_B))
    {
        const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
        temp.set_end_op(clr_unused_unpacr_engine);
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for unpack to SrcA or SrcB with a tile transpose (A -> A^T or B -> B^T).
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for a single operand, default 1 tile of 32x32
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note Does NOT support tiny-tiles
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_transpose_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be p_unpacr::UNP_A or p_unpacr::UNP_B for unpack transpose");

    LLK_ASSERT(tensor_shape.total_num_faces() == NUM_FACES, "Transpose is only supported for regular tile dimensions with 4 faces");

    const std::uint32_t MOP_OUTER_LOOP = num_tiles;
    const std::uint32_t MOP_INNER_LOOP = 1;

    constexpr std::uint32_t replay_buf_len = NUM_FACES;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id]
        {
            if constexpr (UNP_SEL == p_unpacr::UNP_A)
            {
                TT_UNPACR0_FACE(
                    0 /*Dst_Face_Idx*/,
                    0 /*Src_Face_Idx*/,
                    0 /*Dst_Tile_Offset_Idx_Inc*/,
                    0 /*Src_Tile_Offset_Idx_Inc*/,
                    buf_desc_id,
                    0 /*SetDatValid*/); // Unpacks face 0 into dest offset 0
                TT_UNPACR0_FACE(
                    1 /*Dst_Face_Idx*/,
                    2 /*Src_Face_Idx*/,
                    0 /*Dst_Tile_Offset_Idx_Inc*/,
                    0 /*Src_Tile_Offset_Idx_Inc*/,
                    buf_desc_id,
                    0 /*SetDatValid*/); // Unpacks face 2 into dest offset 1
                TT_UNPACR0_FACE(
                    2 /*Dst_Face_Idx*/,
                    1 /*Src_Face_Idx*/,
                    0 /*Dst_Tile_Offset_Idx_Inc*/,
                    0 /*Src_Tile_Offset_Idx_Inc*/,
                    buf_desc_id,
                    0 /*SetDatValid*/); // Unpacks face 1 into dest offset 2
                TT_UNPACR0_FACE(
                    3 /*Dst_Face_Idx*/,
                    3 /*Src_Face_Idx*/,
                    0 /*Dst_Tile_Offset_Idx_Inc*/,
                    0 /*Src_Tile_Offset_Idx_Inc*/,
                    buf_desc_id,
                    1 /*SetDatValid*/); // Unpacks face 3 into dest offset 3
            }
            else if constexpr (UNP_SEL == p_unpacr::UNP_B)
            {
                TT_UNPACR1_FACE(
                    0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0 /*Dst_Tile_Offset_Idx_Inc*/, 0 /*Src_Tile_Offset_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
                TT_UNPACR1_FACE(
                    1 /*Dst_Face_Idx*/, 2 /*Src_Face_Idx*/, 0 /*Dst_Tile_Offset_Idx_Inc*/, 0 /*Src_Tile_Offset_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
                TT_UNPACR1_FACE(
                    2 /*Dst_Face_Idx*/, 1 /*Src_Face_Idx*/, 0 /*Dst_Tile_Offset_Idx_Inc*/, 0 /*Src_Tile_Offset_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
                TT_UNPACR1_FACE(
                    3 /*Dst_Face_Idx*/, 3 /*Src_Face_Idx*/, 0 /*Dst_Tile_Offset_Idx_Inc*/, 0 /*Src_Tile_Offset_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
            }
        });
    ckernel_template temp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_REPLAY(0 /*start_idx*/, replay_buf_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/),
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 1 /*Value*/)); // Inc Src by 1 tile, because above UNPACR0/1_FACE do not inc counters

    // 32-bit datacopy uses ELWADD, which requires datavalid from both SrcA and SrcB
    if constexpr (IS_32b_DEST_EN)
    {
        const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
        temp.set_end_op(clr_unused_unpacr_engine);
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for unpack when reuse_dest is active for eltwise binary operations.
 * @details Uses UNPACR_FACE_INC with per-face NOP dvalid for the MOVD-filled source register.
 * For DEST_TO_SRCA: SrcA is filled by MOVD2A (gets dummy dvalid NOP), SrcB is unpacked from CB.
 * For DEST_TO_SRCB: SrcB is filled by MOVD2B (gets dummy dvalid NOP), SrcA is unpacked from CB.
 * MOP inner loop iterates over NUM_FACES, END_OP advances the tile counter.
 * @tparam UNP_SEL: The original unpack selector (not used directly for reuse_dest dispatch)
 * @tparam reuse_dest: Which source register is reused from dest (DEST_TO_SRCA or DEST_TO_SRCB)
 * @param buf_desc_id: The buffer descriptor ID for the CB source
 * @param num_tiles: number of tiles to unpack
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <std::uint32_t UNP_SEL, EltwiseBinaryReuseDestType reuse_dest>
inline void _llk_unpack_unary_operand_reuse_dest_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    static_assert(reuse_dest != EltwiseBinaryReuseDestType::NONE, "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB");

    // CB_UNP: the unpacker that reads real data from the Circular Buffer
    // DUMMY_UNP: the unpacker that gets a dummy dvalid (its source register is filled by MOVD2A/B on the math side)
    constexpr std::uint32_t CB_UNP    = (reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    constexpr std::uint32_t DUMMY_UNP = (reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? p_unpacr::UNP_A : p_unpacr::UNP_B;

    // Dummy dvalid NOP for the source register filled by MOVD2A/B
    const std::uint32_t nop_op =
        TT_OP_UNPACR_NOP(DUMMY_UNP, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, 0 /*Src_ClrVal_Ctrl*/, p_unpacr::UNP_NOP);

    // Unpack one face from CB with auto-increment of src face index.
    // Dst_Face_Idx_Inc=0: always write to face 0 position (FPU reads from 0 after CLR_AB).
    // Src_Face_Idx_Inc=1: advance through L1 tile faces 0→1→2→3 (wraps back to 0).
    if (tensor_shape.total_num_faces() == NUM_FACES) // Using regular tile dimensions
    {
        const std::uint32_t face_inc_op = (CB_UNP == p_unpacr::UNP_A) ? TT_OP_UNPACR0_FACE_INC(
                                                                            0 /*Dst_Face_Idx_Inc*/,
                                                                            1 /*Src_Face_Idx_Inc*/,
                                                                            0 /*Dst_Tile_Offset_Idx_Inc*/,
                                                                            0 /*Src_Tile_Offset_Idx_Inc*/,
                                                                            buf_desc_id,
                                                                            1 /*SetDatValid*/)
                                                                      : TT_OP_UNPACR1_FACE_INC(
                                                                            0 /*Dst_Face_Idx_Inc*/,
                                                                            1 /*Src_Face_Idx_Inc*/,
                                                                            0 /*Dst_Tile_Offset_Idx_Inc*/,
                                                                            0 /*Src_Tile_Offset_Idx_Inc*/,
                                                                            buf_desc_id,
                                                                            1 /*SetDatValid*/);
        // MOP: outer=num_tiles, inner=num_faces
        // Each inner iteration: NOP (dvalid for dummy src) + FACE_INC (unpack face + inc src face)
        // END_OP: increment CB tile counter after all faces of a tile are processed
        ckernel_template temp(num_tiles, tensor_shape.total_num_faces(), nop_op, face_inc_op);
        temp.set_end_op(TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, CB_UNP, 1 /*Value*/));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else // Using tiny-tiles
    {
        const std::uint32_t face_inc_op = (CB_UNP == p_unpacr::UNP_A)
                                              ? TT_OP_UNPACR0_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/)
                                              : TT_OP_UNPACR1_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*SetDatValid*/);
        // MOP: outer=num_tiles, inner=num_faces
        // Each inner iteration: NOP (dvalid for dummy src) + FACE_INC (unpack face + inc src face)
        ckernel_template temp(num_tiles, tensor_shape.total_num_faces(), nop_op, face_inc_op);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker to unpack a single operand by tiles.
 *
 * Programs the transpose config, then dispatches to the reuse_dest, transpose, or plain MOP builder.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB, values = <true/false>
 * @tparam IS_32b_DEST_EN: Enables using the math destination register in 32-bit mode, values = <true/false>
 * @tparam reuse_dest: When not NONE, configures per-face unpack with dummy dvalid, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: When true, selects the semaphore-synchronized unpack-to-DEST path; requires UNP_SEL == UNP_DEST, values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @param num_tiles: Number of tiles to unpack at a time for a single operand; default 1 tile of 32x32.
 * @note On the math thread (T1): for the plain datacopy path pair with @ref _llk_math_eltwise_unary_datacopy_init_; for reuse_dest != NONE this is the
 *       unpack half of an eltwise binary op, so pair with @ref _llk_math_eltwise_binary_init_ (the dummy-dvalid NOP here feeds the source register that math
 *       fills with MOVD2A/B from dest).
 * @note @ref _llk_unpack_unary_operand_ is the matching execute call on this thread.
 */
template <
    std::uint32_t UNP_SEL,
    bool TRANSPOSE_EN,
    bool IS_32b_DEST_EN,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                   = false>
inline void _llk_unpack_unary_operand_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    static_assert(!(TRANSPOSE_EN && reuse_dest != EltwiseBinaryReuseDestType::NONE), "Transpose is not supported with reuse_dest");

    if constexpr (unpack_to_dest)
    {
        static_assert(UNP_SEL == p_unpacr::UNP_DEST, "unpack_to_dest path requires UNP_SEL == p_unpacr::UNP_DEST");

        // Unpack owns the DEST section base in the unpack-to-dest path: it is the DEST
        // producer (UNP_DEST), so it programs the per-TRISC section base itself rather than
        // letting the math middleman set it on its behalf
        // Establish the initial bank-0 base here; the per-tile call flips
        // it in SyncHalf. TriscID::Unpack selects the same SEC slot the UNP_DEST client reads.
        ckernel::trisc::_reset_dest_register_offset_();
        ckernel::trisc::_set_dest_section_base_<to_underlying(ckernel::trisc::TriscID::Unpack)>(ckernel::trisc::_get_dest_buffer_base_());

        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
        _llk_unpack_unary_operand_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, num_tiles);
        return;
    }

    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
    }

    if constexpr (reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        _llk_unpack_unary_operand_reuse_dest_mop_config_<UNP_SEL, reuse_dest>(buf_desc_id, num_tiles, tensor_shape);
    }
    else if constexpr (TRANSPOSE_EN)
    {
        _llk_unpack_unary_operand_transpose_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, num_tiles, tensor_shape);
    }
    else
    {
        if constexpr (UNP_SEL == p_unpacr::UNP_DEST) // workaround for unpack to dest, since not supported for tiny tile currently
        {
            _llk_unpack_unary_operand_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, num_tiles);
        }
        else
        {
            if (tensor_shape.total_num_faces() == NUM_FACES || tensor_shape.total_num_faces() == 1) // Using regular tile dimensions
            {
                _llk_unpack_unary_operand_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, num_tiles);
            }
            else // Using tiny-tiles
            {
                _llk_unpack_unary_operand_variable_tile_size_mop_config_<UNP_SEL, IS_32b_DEST_EN>(buf_desc_id, num_tiles, tensor_shape);
            }
        }
    }
}

/**
 * @brief Unpacks a single operand; works for any unpack resource.
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = <p_unpacr::UNP_A/UNP_B/UNP_DEST>
 * @tparam reuse_dest: When not NONE, sets the source counter for the CB unpacker only, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: When true, runs the UNPACK_MATH/MATH_PACK semaphore handshake for unpack-to-DEST; requires UNP_SEL == UNP_DEST, values = <true/false>
 * @tparam DEST_SYNC_MODE: In the unpack-to-DEST path, SyncHalf flips the DEST section base to the other bank after each tile, values = <SyncFull/SyncHalf>
 * @param l1_tile_idx: Index into the L1 buffer for a tile.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note Call @ref _llk_unpack_unary_operand_init_ with matching template args before this function.
 */
template <
    std::uint32_t UNP_SEL,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                   = false,
    ckernel::DstSync DEST_SYNC_MODE       = ckernel::DstSync::SyncFull>
inline void _llk_unpack_unary_operand_(const std::uint32_t l1_tile_idx, const TensorShape& tensor_shape)
{
    if constexpr (unpack_to_dest)
    {
        static_assert(UNP_SEL == p_unpacr::UNP_DEST, "unpack_to_dest path requires UNP_SEL == p_unpacr::UNP_DEST");

        // The math thread is the middleman with two single-counting semaphores (max=N each).
        // Without an extra wait on MATH_PACK, unpack could race 2N iterations ahead of pack
        // and overwrite a bank that pack has not read yet. Waiting on both keeps unpack
        // within N iterations of pack.
        _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);

        // UNP_DEST is driven off the UNP_A bank's counters.
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

        // Drain UNPACK0 before posting "filled" so the post does not race the writes math reads.
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);

        // Unpack owns the DEST section base, so it flips to the other bank for the next iteration
        if constexpr (DEST_SYNC_MODE == ckernel::DstSync::SyncHalf)
        {
            _llk_sync_advance_dest_section_<to_underlying(ckernel::trisc::TriscID::Unpack), true /*EN_32BIT_DEST*/, p_stall::UNPACK0>();
        }
        return;
    }

    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    if constexpr (reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        // For reuse_dest, set source counter for the unpacker that reads from the Circular Buffer.
        // The other source register is filled by MOVD2A/B on the math side.
        // For tiny-tiles, each face is considered a separate tile in HW. We need to multiply the tile idx by num_faces to get the correct SW defined tile
        // offset.
        constexpr std::uint32_t CB_UNP = (reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
        TT_SET_SRC_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL, CB_UNP, (tensor_shape.total_num_faces() == NUM_FACES) ? l1_tile_idx : l1_tile_idx * tensor_shape.total_num_faces());
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, CB_UNP, 0 /*Value*/);
    }
    else
    {
        // Reset Dest counters for Unpacker to 0
        // Set Source counter to L1 base + offset
        // For tiny-tiles, each face is considered a separate tile in HW. We need to multiply the tile idx by num_faces to get the correct SW defined tile
        // offset.
        TT_SET_SRC_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL,
            UNP_SEL == p_unpacr::UNP_DEST ? p_unpacr::UNP_A : UNP_SEL,
            (tensor_shape.total_num_faces() == NUM_FACES) ? l1_tile_idx : l1_tile_idx * tensor_shape.total_num_faces());
        TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL == p_unpacr::UNP_DEST ? p_unpacr::UNP_A : UNP_SEL, 0 /*Value*/);
    }

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
