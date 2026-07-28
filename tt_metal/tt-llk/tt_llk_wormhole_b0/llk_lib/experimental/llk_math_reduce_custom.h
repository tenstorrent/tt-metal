// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "lltt.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * Configures address modifiers for specialized reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block/tile operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * @note ADDR_MOD_3 (the +20 DEST jump into F2) is only consumed by the two-face-row
 *       (num_faces=4) code path. For num_faces=2 there is no F2/F3, so it is unused, but it
 *       is still programmed here so that both code paths share one addrmod configuration.
 *
 * This function should NOT be used as a substitute for native reduce LLK configuration.
 * Use the standard reduce configuration functions for general-purpose reduction operations.
 */
inline void reduce_max_row_configure_addrmod()
{
    // ADDR_MOD_0: Default mode used with pool operations (GMPOOL/GAPOOL)
    // No auto-increment on any counter - keeps all address pointers at their current positions
    // Used for initial pooling operations where we want explicit control over counter advancement
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_0);

    // ADDR_MOD_1: Face-to-face advancement mode used with GMPOOL operations
    // Increments SrcA by 16 rows to advance to the next face (F0->F1, F2->F3)
    // Clears SrcB counter to prepare for subsequent MOVD2B operations that write transposed results
    // This allows processing pairs of faces (F0+F1, then F2+F3) efficiently
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_2: Sequential 4-row write mode used with MOVB2D operations
    // Increments DEST by 4 rows after each operation (writes to rows 0, 4, 8, 12 within a face)
    // This distributes the transposed 1x16 reduction result across the first face (rows 0-15)
    // by writing 4 rows to every 4th row, matching the tile face layout
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);

    // ADDR_MOD_3: Face-boundary jump mode used with final MOVB2D in each face
    // Increments DEST by 20 rows, jumping from row 12 to row 32 (= 12 + 4 + 16)
    // This transitions from the last row of the current face to the first row of the next face
    // Example: After writing rows 0,4,8,12 in F0, this jumps to row 32 to start F2
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 20, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);
}

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * For num_faces=4 the per-tile MOP reduces F0&F1 into DEST rows 0-15, jumps DEST to row 32 via a
 * CR_D+8 SETRWC repeated inner_loop=4 times (4*8=32 = row 32 = F2), then reduces F2&F3, and the
 * transpose (replayed by the execute function) covers both face-rows.
 *
 * For num_faces=2 there is only one face-row (F0&F1): the per-tile MOP reduces F0&F1 only, with no
 * F2 jump and no second reduce, and the recorded transpose block covers the single face-row.
 *
 * This function should NOT be used as a substitute for native reduce LLK MOP configuration.
 * Use the standard reduce MOP configuration with _llk_math_reduce_init_ for general-purpose reduction.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape& tensor_shape)
{
    // Constraint on the outerloop and innerloop dim
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row (16x32 tiny tile): only F0&F1 exist. Reduce them, transpose once, no F2 jump.
        if constexpr (is_fp32_dest_acc_en)
        {
            // FP32 path records the same 15-instruction layout as num_faces=4 (2 GMPOOLs + hi/lo
            // transpose + CLR_B). Only the execute-side replay differs (single face-row, done below).
            lltt::record(0, 15);

            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

            // Move high 16 bits from DEST row 0 to SrcB rows 16 - 31 and transpose
            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // Move high 16 bits back to Dest
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 4);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 8);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);
            // Move low 16 bits to SrcB rows 16 - 31 and transpose
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // Move low 16 bits from SrcB rows 16 - 31 to DEST rows 0, 4, 8, 12.
            // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so DEST location is '0', not '0, 4, 8, 12'.
            // No F2 exists for a single face-row, so the last write uses ADDR_MOD_2 (no +20 jump); CLR_B
            // resets the counters afterwards regardless.
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            // Clear B valid bits at the end and all address counters
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }
        else
        {
            // Non-FP32 single face-row: 2 GMPOOLs + one transpose block (6 instrs) + CLR_B = 9 instructions.
            lltt::record(0, 9);

            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

            // Move row 0 from DEST to SrcB with offset of 16 rows and transpose
            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // Move the reduced row from SrcB to DEST in 4-row chunks (rows 0, 4, 8, 12).
            // ADDR_MOD_2 increments CR_D and Dest counter val by 4. No F2 jump (no ADDR_MOD_3): the
            // last write uses ADDR_MOD_2 and CLR_B resets the counters afterwards.
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            // Clear B valid bits at the end and all address counters
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }

        static constexpr std::uint32_t outer_loop = block_ct_dim;
        static constexpr std::uint32_t inner_loop = 1;

        // Reduce F0 and F1 column-wise, doesn't change DEST counter
        static constexpr std::uint32_t start_op = TT_OP_REPLAY(0, 2, 0, 0);
        // No F2 jump for a single face-row
        static constexpr std::uint32_t inner_loop_op = TT_OP_NOP;
        // Clear A valid bit to get another operand tile (scaler face stays the same)
        static constexpr std::uint32_t end_op_1 = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
        static constexpr std::uint32_t end_op_2 = TT_OP_NOP;

        ckernel::ckernel_template mop_template(outer_loop, inner_loop, inner_loop_op);
        mop_template.set_start_op(start_op);
        mop_template.set_end_ops(end_op_1, end_op_2);
        mop_template.program();
        return;
    }

    // See _llk_math_reduce_max_row_ for a full algorithm explanation
    // Put the following 15 instructions in a REPLAY buffer
    lltt::record(0, 15);

    // Two GMPOOLs to pool F0 and F1 (or F2 and F3) together
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

    // The following instructions are going to transpose the whole tile.

    // Move row 0 from DEST to SrcB with offset of 16 rows and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    // Move row 0 from SrcB to DEST in 4-row chunks
    // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so that's why DEST location is '0', not '0, 4, 8, 12'.
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    // ADDR_MOD_3 increments CR_D and Dest counter val by 20, to point to F2.
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);

    // Move row 32 (F2R0) from DEST to SrcB with offset of 16 rows and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    // Move row 32 from SrcB to DEST in 4-row chunks
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 4);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 8);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

    // Clear B valid bits at the end and all address counters
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);

    static constexpr std::uint32_t outer_loop = block_ct_dim;
    static constexpr std::uint32_t inner_loop = 4;

    // Reduce F0 and F1 column-wise, doesn't change DEST counter
    static constexpr std::uint32_t start_op = TT_OP_REPLAY(0, 2, 0, 0);
    // Increment DEST counter by 32 to point to F2 (DEST row 32)
    static constexpr std::uint32_t inner_loop_op = TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    // Reduce F2 and F3 column-wise, doesn't change DEST counter, but clears A valid bits at the end
    static constexpr std::uint32_t end_op_1 = TT_OP_REPLAY(0, 2, 0, 0);
    // Clear A valid bit to get another operand tile (scaler face stays the same)
    static constexpr std::uint32_t end_op_2 = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);

    ckernel::ckernel_template mop_template(outer_loop, inner_loop, inner_loop_op);
    mop_template.set_start_op(start_op);
    mop_template.set_end_ops(end_op_1, end_op_2);
    mop_template.program();
}

/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_init_ LLK.
 * Use the standard _llk_math_reduce_init_<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * _llk_math_reduce_() calls in a loop for general-purpose block reduction.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_ LLK.
 * Use the standard _llk_math_reduce_<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    // Packer indexes at the 32x32 slot stride regardless of the operand's face count.
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Run the MOP. For num_faces=4 this reduces both face-rows (F0&F1 then F2&F3); for num_faces=2
    // it reduces the single face-row (F0&F1). The MOP only emits the GMPOOLs; the transpose is
    // replayed below.
    ckernel::ckernel_template::run();

    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row: transpose only the one recorded face-row. The 2-face record has no
        // ADDR_MOD_3 F2 jump, so no spurious DEST advance occurs.
        if constexpr (is_fp32_dest_acc_en)
        {
            // MOVB2D/D2B depends on the SrcA ALU format; Hi/Lo16 transpose does not work with the
            // Tf32 that the pool wrote, so override SrcA to Tf32 (and disable the zero-flag source)
            // for the replayed transpose, exactly as the num_faces=4 path does below.
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));

            // Replay the 13 instructions (recorded slots 2-14) to transpose the single reduced
            // face-row: hi/lo transpose + CLR_B.
            lltt::replay(2, 13);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
        }
        else
        {
            // Replay the 7 instructions (recorded slots 2-8) to transpose the single reduced
            // face-row. 7th instruction (CLR_B) releases SrcB bank and clears all address counters.
            lltt::replay(2, 7);
        }
        return;
    }

    if constexpr (is_fp32_dest_acc_en)
    {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));
    }

    // Replay the 13 instructions to transpose the reduced results
    lltt::replay(2, 13);

    if constexpr (is_fp32_dest_acc_en)
    {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    }
}
