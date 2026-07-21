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
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce LLK configuration.
 * Use the standard reduce configuration functions for general-purpose reduction operations.
 */
inline void reduce_max_row_configure_addrmod_runtime()
{
    // ADDR_MOD_6: Default mode used with pool operations (GMPOOL/GAPOOL)
    // No auto-increment on any counter - keeps all address pointers at their current positions
    // Used for initial pooling operations where we want explicit control over counter advancement
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_6);

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

inline void reduce_max_row_configure_addrmod_reinit_runtime()
{
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 20, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);
}

// Minimal reinit: ADDR_MOD_1 + ADDR_MOD_2 (clobbered by matmul) and ADDR_MOD_6
// (clobbered by sub_exp runtime). Skips ADDR_MOD_3 which is preserved from the
// previous reduce (matmul doesn't touch 3, sub_exp custom doesn't touch 3,
// copy_tile_custom_v2 doesn't touch 3).
inline void reduce_max_row_configure_addrmod_reinit_minimal_runtime()
{
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);
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
 * For num_faces=2 the per-tile MOP reduces only F0&F1 (no F2 jump, no second reduce) and the
 * recorded transpose block covers the single face-row.
 *
 * This function should NOT be used as a substitute for native reduce LLK MOP configuration.
 * Use the standard reduce MOP configuration with _llk_math_reduce_init_ for general-purpose reduction.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    // Constraint on the outerloop and innerloop dim
    // static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row (16x32 tiny tile): only F0&F1 exist. Reduce them, transpose once, no F2 jump.
        if constexpr (is_fp32_dest_acc_en)
        {
            // FP32 path records the same 15-instruction layout; only the execute-side replay differs.
            lltt::record(0, 15);

            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 4);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 8);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 12);
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // No F2 for a single face-row: last write uses ADDR_MOD_2 (no +20 jump); CLR_B resets counters.
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }
        else
        {
            // Non-FP32 single face-row: 2 GMPOOLs + one transpose block (6 instrs) + CLR_B = 9 instructions.
            lltt::record(0, 9);

            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }

        const std::uint32_t outer_loop_2f               = block_ct_dim;
        const std::uint32_t inner_loop_2f               = 1;
        static constexpr std::uint32_t start_op_2f      = TT_OP_REPLAY(0, 2, 0, 0);
        static constexpr std::uint32_t inner_loop_op_2f = TT_OP_NOP;
        static constexpr std::uint32_t end_op_1_2f      = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
        static constexpr std::uint32_t end_op_2_2f      = TT_OP_NOP;

        ckernel::ckernel_template mop_template(outer_loop_2f, inner_loop_2f, inner_loop_op_2f);
        mop_template.set_start_op(start_op_2f);
        mop_template.set_end_ops(end_op_1_2f, end_op_2_2f);
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
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    // Move row 0 from SrcB to DEST in 4-row chunks
    // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so that's why DEST location is '0', not '0, 4, 8, 12'.
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    // ADDR_MOD_3 increments CR_D and Dest counter val by 20, to point to F2.
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);

    // Move row 32 (F2R0) from DEST to SrcB with offset of 16 rows and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    // Move row 32 from SrcB to DEST in 4-row chunks
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 4);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 8);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 12);

    // Clear B valid bits at the end and all address counters
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);

    const std::uint32_t outer_loop = block_ct_dim;
    const std::uint32_t inner_loop = 4;

    static constexpr std::uint32_t start_op      = TT_OP_REPLAY(0, 2, 0, 0);
    static constexpr std::uint32_t inner_loop_op = TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    static constexpr std::uint32_t end_op_1      = TT_OP_REPLAY(0, 2, 0, 0);
    static constexpr std::uint32_t end_op_2      = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);

    ckernel::ckernel_template mop_template(outer_loop, inner_loop, inner_loop_op);
    mop_template.set_start_op(start_op);
    mop_template.set_end_ops(end_op_1, end_op_2);
    mop_template.program();
}

/**
 * Reprograms only the MOP registers for reduce_max_row, without re-recording the replay buffer.
 * The replay buffer at positions 0-14 must still contain the reduce GMPOOL+transpose sequence
 * from the original _llk_math_reduce_block_max_row_mop_config_ call.
 * Use when the MOP was clobbered (e.g., by eltwise binary ops) but the replay buffer is intact.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_reprogram_only_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row: reduce F0&F1 only, no F2 jump, no second reduce.
        const std::uint32_t outer_loop_2f               = block_ct_dim;
        const std::uint32_t inner_loop_2f               = 1;
        static constexpr std::uint32_t start_op_2f      = TT_OP_REPLAY(0, 2, 0, 0);
        static constexpr std::uint32_t inner_loop_op_2f = TT_OP_NOP;
        static constexpr std::uint32_t end_op_1_2f      = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
        static constexpr std::uint32_t end_op_2_2f      = TT_OP_NOP;

        ckernel::ckernel_template mop_template(outer_loop_2f, inner_loop_2f, inner_loop_op_2f);
        mop_template.set_start_op(start_op_2f);
        mop_template.set_end_ops(end_op_1_2f, end_op_2_2f);
        mop_template.program();
        return;
    }

    const std::uint32_t outer_loop = block_ct_dim;
    const std::uint32_t inner_loop = 4;

    static constexpr std::uint32_t start_op      = TT_OP_REPLAY(0, 2, 0, 0);
    static constexpr std::uint32_t inner_loop_op = TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    static constexpr std::uint32_t end_op_1      = TT_OP_REPLAY(0, 2, 0, 0);
    static constexpr std::uint32_t end_op_2      = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);

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
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod_runtime();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_runtime_()
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
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_runtime_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
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
            TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

            // Replay the 13 instructions (slots 2-14): hi/lo transpose of the single face-row + CLR_B.
            lltt::replay(2, 13);

            TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
        }
        else
        {
            // Replay the 7 instructions (slots 2-8) to transpose the single reduced face-row + CLR_B.
            lltt::replay(2, 7);
        }
        return;
    }

    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
    }

    // Replay the 13 instructions to transpose the reduced results
    lltt::replay(2, 13);

    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    }
}

/**
 * Reinitializes the block-based reduce_max_row operation after a matmul.
 *
 * This LLK API function is used only to re-initialize the address modifiers after a
 * matmul operation in an SDPA inner loop. Please don't use this function as a substitute for
 * the native llk_math_reduce_block_max_row_init LLK. This function is highly specialized
 * for a certain use case and the LLK team does not guarantee any degree of generality.
 */
inline void _llk_math_reduce_block_max_row_reinit_runtime_()
{
    reduce_max_row_configure_addrmod_reinit_runtime();
}

/**
 * Short reinitialization for block-based reduce_max_row operation after a matmul.
 * Reprograms address modifiers and MOP configuration. Used when MOP was clobbered
 * but full init is not needed.
 *
 * This LLK API function is used only to re-initialize the address modifiers and MOP
 * after a matmul operation in an SDPA inner loop.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_short_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_reduce_block_max_row_mop_reprogram_only_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * Minimal reinitialization for block-based reduce_max_row operation.
 * Only reconfigures ADDR_MOD_1, ADDR_MOD_2, and ADDR_MOD_6 (preserves ADDR_MOD_3).
 * Used when only specific addrmods were clobbered by previous operations.
 */
inline void _llk_math_reduce_block_max_row_reinit_minimal_runtime_()
{
    reduce_max_row_configure_addrmod_reinit_minimal_runtime();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
