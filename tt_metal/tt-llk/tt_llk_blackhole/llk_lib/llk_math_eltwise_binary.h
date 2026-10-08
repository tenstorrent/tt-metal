// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

/*************************************************************************
 * Common Helpers
 *************************************************************************/
/**
 * @brief Program the four address-mod slots used by eltwise binary MOPs (per-row, no-op, fidelity-step, face-step).
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam tile_dvalid: Whole-tile program (see @ref eltwise_binary_tile_dvalid): ADDR_MOD_2 and ADDR_MOD_3 return the counters to the tile base
 * @tparam row_step: Slot of the whole-tile row broadcast's face step; the dest-reuse execute keeps ADDR_MOD_1 for its moves and clears
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    MathFidelity math_fidelity,
    bool tile_dvalid      = false,
    std::uint8_t row_step = ADDR_MOD_1>
inline void eltwise_binary_configure_addrmod()
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr           = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? MAX_FPU_ROWS : 0;
    addr_mod_t {
        .srca = {.incr = MAX_FPU_ROWS},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = MAX_FPU_ROWS},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    if constexpr (tile_dvalid)
    {
        if constexpr (bcast_type == BroadcastType::ROW)
        {
            // The second instruction of a face moves SrcB to the next face's broadcast row
            addr_mod_t {
                .srca = {.incr = MAX_FPU_ROWS},
                .srcb = {.incr = FACE_R_DIM},
                .dest = {.incr = MAX_FPU_ROWS},
            }
                .set(row_step);
        }

        addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = fidelity_increment}}
            .set(ADDR_MOD_2);

        addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_3);
        return;
    }

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = fidelity_increment}}
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1},
        .srcb     = {.incr = 0, .clr = 1},
        .dest     = {.incr = MAX_FPU_ROWS, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
}

/**
 * @brief Whether the dest-reuse path consumes each operand tile as one source bank: SrcDvalid::PerTile without a broadcast, or with a row
 *        broadcast of the L1 operand (DEST_TO_SRCA); it does for full 16-row faces, 2 x 2 of them for the row broadcast. The dest-reuse
 *        unpack init applies the same rule (@ref unpack_A_tile_dvalid), so the two threads agree.
 */
template <BroadcastType bcast_type, EltwiseBinaryReuseDestType binary_reuse_dest, SrcDvalid src_dvalid>
inline constexpr bool eltwise_binary_tile_dvalid =
    src_dvalid == SrcDvalid::PerTile &&
    (bcast_type == BroadcastType::NONE || (bcast_type == BroadcastType::ROW && binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA));

/**
 * @brief Whether the standard (two-operand) path consumes each operand tile as one source bank: SrcDvalid::PerTile, with or without a
 *        broadcast, for the shapes of @ref eltwise_binary_tile_shape. The two-operand unpack init applies the same rule (@ref unpack_AB_tile_dvalid).
 */
template <SrcDvalid src_dvalid>
inline constexpr bool eltwise_binary_standard_tile_dvalid = src_dvalid == SrcDvalid::PerTile;

/**
 * @brief Whether a column broadcast tile takes the program of @ref eltwise_binary_configure_tile_partial_col: two partial faces
 *        (face_r_dim 1 to 8) side by side. The two-operand unpack init applies the same rule.
 */
template <BroadcastType bcast_type>
inline bool eltwise_binary_tile_partial_col_shape(const ckernel::TensorShape tensor_shape)
{
    return bcast_type == BroadcastType::COL && tensor_shape.face_r_dim < FACE_R_DIM && tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2;
}

/**
 * @brief Whether a tile takes the whole-tile program: full 16-row faces, and 2 x 2 faces for a column or row broadcast (whose SrcB bank
 *        holds B's faces in the order the unpack init writes them, 0 0 2 2 for a column broadcast and 0 1 0 1 for a row broadcast);
 *        and the column broadcast tiles of @ref eltwise_binary_tile_partial_col_shape.
 */
template <BroadcastType bcast_type>
inline bool eltwise_binary_tile_shape(const ckernel::TensorShape tensor_shape)
{
    constexpr bool needs_2x2_faces = bcast_type == BroadcastType::COL || bcast_type == BroadcastType::ROW;
    return (tensor_shape.face_r_dim == FACE_R_DIM && (!needs_2x2_faces || (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 2))) ||
           eltwise_binary_tile_partial_col_shape<bcast_type>(tensor_shape);
}

/**
 * @brief Whether the dest-reuse path takes the whole-tile program for this tile shape (see @ref eltwise_binary_tile_dvalid): full 16-row faces,
 *        2 x 2 of them for a row broadcast. The dest-reuse unpack init applies the same rule.
 */
template <BroadcastType bcast_type>
inline bool eltwise_binary_reuse_tile_shape(const ckernel::TensorShape tensor_shape)
{
    return tensor_shape.face_r_dim == FACE_R_DIM &&
           (bcast_type != BroadcastType::ROW || (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 2));
}

/**
 * @brief Build the encoded FPU instruction (ELWADD/ELWSUB/ELWMUL) for the given binary op type.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @param clr_src: Source-clear mode passed to the instruction.
 * @param acc_to_dest: Accumulate result into dest instead of overwriting.
 * @param broadcast_type: Source B broadcast mode (p_elwise::SRCB_* value).
 * @param addr_mod: Address-mod slot the instruction uses.
 * @return The encoded TT_OP instruction word.
 */
template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
    {
        return TT_OP_ELWADD(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
    {
        return TT_OP_ELWSUB(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
    else
    {
        return TT_OP_ELWMUL(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
}

/**
 * @brief Configure the MOP that processes a whole tile from one source bank per operand (see @ref eltwise_binary_tile_dvalid):
 *        num_faces x 2 eight-row instructions per fidelity phase, both source banks released once at the end.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR> (see @ref eltwise_binary_tile_shape for the SrcB layout)
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam row_step: Address-mod slot of a row broadcast's face step, as programmed by @ref eltwise_binary_configure_addrmod
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting (ELWADD/ELWSUB only)
 * @param num_faces: Faces of the tile, 1, 2 or 4
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity, std::uint8_t row_step = ADDR_MOD_1>
inline void eltwise_binary_configure_mop_tile(const std::uint32_t acc_to_dest, const std::uint32_t num_faces)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;
    // A row broadcast pairs the two instructions of a face: ADDR_MOD_0 keeps SrcB on the face's row, row_step moves it to the next face
    constexpr bool row_pairs      = bcast_type == BroadcastType::ROW;
    const std::uint32_t innerloop = row_pairs ? num_faces : num_faces * (FACE_R_DIM >> MAX_FPU_ROWS_LOG2);
    const std::uint32_t acc       = (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? 0 : acc_to_dest;

    if constexpr (is_high_fidelity(math_fidelity))
    {
        const std::uint32_t elwmul = eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_0);
        if constexpr (row_pairs)
        {
            ckernel_template tmp(
                to_underlying(math_fidelity),
                innerloop,
                elwmul,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, row_step));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(p_setrwc::CLR_AB, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(to_underlying(math_fidelity), innerloop, elwmul);
            tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(p_setrwc::CLR_AB, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));
            tmp.program();
        }
    }
    else if constexpr (row_pairs)
    {
        ckernel_template tmp(
            1,
            innerloop,
            eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc, broadcast_type, ADDR_MOD_0),
            eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc, broadcast_type, row_step));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, innerloop, eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc, broadcast_type, ADDR_MOD_0));
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
}

/**
 * @brief Configure the address mods and the MOP of the whole-tile program for a column broadcast tile of two partial faces (see
 *        @ref eltwise_binary_tile_partial_col_shape): SrcA holds each face in its own 16-row slot and SrcB holds B's face 0; per
 *        fidelity phase one eight-row instruction per face, both source banks released once at the end.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting (ELWADD/ELWSUB only)
 */
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_tile_partial_col(const std::uint32_t acc_to_dest)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr auto broadcast_type              = p_elwise::SRCB_BCAST_COL;
    const std::uint32_t acc                    = (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? 0 : acc_to_dest;

    addr_mod_t {
        .srca = {.incr = FACE_R_DIM},
        .srcb = {.incr = 0},
        .dest = {.incr = FACE_R_DIM},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = fidelity_increment}}.set(
        ADDR_MOD_2);

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_3);

    const std::uint32_t op = eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc, broadcast_type, ADDR_MOD_0);
    if constexpr (is_high_fidelity(math_fidelity))
    {
        ckernel_template tmp(to_underlying(math_fidelity), 1, op, op);
        tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
        tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(p_setrwc::CLR_AB, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, 1, op, op);
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
}

/*************************************************************************
 * Eltwise Binary Standard (No Dest Reuse)
 * Simple: SrcA op SrcB -> Dest
 *************************************************************************/

/**
 * @brief Configure MOP for standard eltwise binary operations (no dest reuse).
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 * @param tensor_shape: Tensor shape describing tile dimensions
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_binary_configure_mop_standard", tensor_shape);
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod     = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    // Even if face_r_dim < 16, we still process at least 1 inner loop iteration
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // Outer loop depends on broadcast type:
    // - COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
    //                  Runtime calls MOP num_faces_r_dim times (one call per row)
    // - Other broadcasts: MOP processes all num_faces in one call
    const std::uint32_t outerloop = (bcast_type == BroadcastType::COL) ? num_faces_c_dim : num_faces;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within a MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc_to_dest, broadcast_type, addr_mod));
        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // For partial faces (face_r_dim < 16), we still need to increment counters by MAX_FPU_ROWS
            // to maintain proper 16-row spacing between faces
            tmp.set_loop_op1(TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
        }
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            ckernel_template tmp(
                to_underlying(math_fidelity),
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(CLR_SRC, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));
            // HiFi partial face advancement is handled by runtime INCRWC between face MOP runs,
            // NOT by end_op (which would incorrectly advance dest between fidelity phases)
            tmp.program();
        }
        else if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // Partial faces: INCRWC as loop_op1 via two-arg constructor to maintain 16-row face spacing.
            // Must use two-arg constructor so m_loop0/1_last_instr = INCRWC, preventing the
            // last-iteration override from replacing INCRWC with a second ELWMUL instruction.
            ckernel_template tmp(
                outerloop,
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod),
                TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(
                outerloop, innerloop, eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse).
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the unpack init
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 * @note @ref _llk_math_eltwise_binary_standard_ runs the configured op with matching template args.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_init_", tensor_shape);

    if constexpr (eltwise_binary_standard_tile_dvalid<src_dvalid>)
    {
        if (eltwise_binary_tile_partial_col_shape<src_b_bcast_type>(tensor_shape))
        {
            eltwise_binary_configure_tile_partial_col<eltwise_binary_type, math_fidelity>(acc_to_dest);
        }
        else if (eltwise_binary_tile_shape<src_b_bcast_type>(tensor_shape))
        {
            eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity, true>();
            eltwise_binary_configure_mop_tile<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape.total_num_faces());
        }
        else
        {
            eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
            eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
        }
    }
    else
    {
        eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
        eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Perform standard elementwise binary operation (no dest reuse): Output = SrcA [+, -, *] SrcB.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the init
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 * @note Call @ref _llk_math_eltwise_binary_standard_init_ with matching template args before this function.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi,
    SrcDvalid src_dvalid       = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_", tensor_shape);
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (eltwise_binary_standard_tile_dvalid<src_dvalid>)
    {
        if (eltwise_binary_tile_shape<src_b_bcast_type>(tensor_shape))
        {
            ckernel_template::run();
            math::clear_dst_reg_addr();
            return;
        }
    }

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
            // Runtime calls MOP num_faces_r_dim times (once per row of faces)
            // After each row, CLR_B to allow next B column to be loaded
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces in one call
            ckernel_template::run();
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes fidelity phases for one face (HiFi) or all face columns (LoFi)
            // With high fidelity, call MOP once per face column per face row
            const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
            const std::uint32_t fidelity_loop   = high_fidelity ? num_faces_c_dim : 1;
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t i = 0; i < fidelity_loop; i++)
                {
                    ckernel_template::run();
                    if constexpr (high_fidelity)
                    {
                        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                        {
                            TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0);
                        }
                    }
                    // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
            {
                ckernel_template::run();
                if constexpr (high_fidelity)
                {
                    if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                    {
                        TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0);
                    }
                }
                // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

/*************************************************************************
 * Eltwise Binary WITH Dest Reuse
 * Complex: Read dest -> Move to src -> Compute -> Store
 *************************************************************************/

/**
 * @brief Move one face of the destination register into a source register (SrcA or SrcB) for dest-reuse ops.
 *
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB>
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        move_d2a_fixed_face(ADDR_MOD_1);
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        move_d2b_fixed_face(ADDR_MOD_1);
    }
}

/**
 * @brief Move one face of the DEST tile into rows 16 x face .. 16 x face + 15 of the reused source bank (four 4-row moves, no wait).
 *
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam face: Face of the tile, 0 to 3 (a template parameter: TTI_ operands must be immediates)
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest, std::uint32_t face>
inline void eltwise_binary_move_dest_face_to_src()
{
    constexpr std::uint32_t row = face * FACE_R_DIM;
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + row + 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, row + 0);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + row + 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, row + 4);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + row + 8, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, row + 8);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + row + 12, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, row + 12);
    }
    else
    {
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + row + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, row + 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + row + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, row + 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + row + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, row + 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + row + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, row + 12);
    }
}

/**
 * @brief Move every face of the DEST tile into the reused source bank with one pipeline drain and bank wait (whole-tile program).
 *
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB>
 * @param num_faces: Faces of the tile, 1, 2 or 4
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src_tile(const std::uint32_t num_faces)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
    }
    eltwise_binary_move_dest_face_to_src<binary_reuse_dest, 0>();
    if (num_faces > 1)
    {
        eltwise_binary_move_dest_face_to_src<binary_reuse_dest, 1>();
    }
    if (num_faces > 2)
    {
        eltwise_binary_move_dest_face_to_src<binary_reuse_dest, 2>();
        eltwise_binary_move_dest_face_to_src<binary_reuse_dest, 3>();
    }
}

/**
 * @brief Clear one 16-row DEST face using ZEROACC's one-row mode.
 *
 * Rows are spelled out instead of looped because TTI_ZEROACC's operand must constant-fold; a loop
 * would only assemble when the unroll hint is honoured, making legality depend on the optimiser.
 * Same reason move_d2a_row_broadcast_fixed_face writes out its row moves.
 */
inline void zeroacc_face_by_row()
{
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, ADDR_MOD_1, 0);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 1);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 2);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 3);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 4);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 5);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 6);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 7);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 8);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 9);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 10);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 11);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 12);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 13);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 14);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 15);
}

/**
 * @brief Configure MOP for eltwise binary operations with dest reuse.
 *
 * MOP outer loop = 1 face, called multiple times externally with ZEROACC between calls.
 * This processes one face at a time because we need to clear dest before each face computation.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param acc_to_dest: Accumulate result to destination register
 * @param tensor_shape: Tensor shape describing tile dimensions
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_binary_configure_mop_with_dest_reuse", tensor_shape);
    constexpr bool high_fidelity    = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // For dest reuse: MOP processes 1 face at a time (outer loop = 1)
    // Runtime calls MOP multiple times with move_d2a/d2b + ZEROACC between calls
    constexpr std::uint32_t outerloop = 1;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc_to_dest, broadcast_type, addr_mod));
        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // For partial faces, still increment by MAX_FPU_ROWS to maintain 16-row face spacing
            tmp.set_loop_op1(TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
        }
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            ckernel_template tmp(
                to_underlying(math_fidelity),
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(CLR_SRC, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));

            tmp.program();
        }
        else if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // Partial faces: INCRWC as loop_op1 via two-arg constructor to maintain 16-row face spacing.
            ckernel_template tmp(
                outerloop,
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod),
                TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(
                outerloop, innerloop, eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize FPU for elementwise binary operations with dest reuse.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB> (NONE is rejected)
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the unpack init
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register
 * @note @ref _llk_math_eltwise_binary_with_dest_reuse_ runs the configured op with matching template args.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA,
    SrcDvalid src_dvalid                         = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_init_ for no dest reuse");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_with_dest_reuse_init_", tensor_shape);

    if constexpr (eltwise_binary_tile_dvalid<src_b_bcast_type, binary_reuse_dest, src_dvalid>)
    {
        if (eltwise_binary_reuse_tile_shape<src_b_bcast_type>(tensor_shape))
        {
            // The execute's moves and clears use ADDR_MOD_1, so a row broadcast steps its faces with ADDR_MOD_4
            constexpr std::uint8_t row_step = (src_b_bcast_type == BroadcastType::ROW) ? ADDR_MOD_4 : ADDR_MOD_1;
            eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity, true, row_step>();
            eltwise_binary_configure_mop_tile<eltwise_binary_type, src_b_bcast_type, math_fidelity, row_step>(acc_to_dest, tensor_shape.total_num_faces());
        }
        else
        {
            eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
            eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
        }
    }
    else
    {
        eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
        eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Run the dest-reuse MOP once per face, moving dest into a source register and zeroing the dest face first.
 *
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register (halves tiles per bank and gates the zero-flag clear).
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param loop_count: Number of faces to process.
 * @param face_offset: Index of the first face within the tile.
 * @param clear_fp32_dst_acc: Clear the FP32 dest accumulator face when FP32 mode is enabled.
 * @param dst_index: Tile index into the destination register.
 * @param face_r_dim: Face row dimension; used for HiFi partial-face dest spacing.
 */
template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset,
    const bool clear_fp32_dst_acc,
    const std::uint32_t dst_index,
    const std::uint32_t face_r_dim)
{
    constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;
    // DEST rows one face spans -- what ZEROACC's 16-row mode clears in a single instruction.
    constexpr std::uint32_t DEST_ROWS_PER_FACE = FACE_R_DIM;
    // DEST rows one tile slot spans. This is DEST geometry, not the tile's logical face count: every tile gets
    // the same slot whatever its shape, because set_dst_write_addr<Tile32x32> places tiles at exactly this stride
    // and the MOP pads partial faces to 16-row spacing (eltwise_binary_configure_mop_with_dest_reuse). A tile with
    // fewer faces uses the low faces of its slot and leaves the rest unused. Derived from the same shift
    // set_dst_write_addr uses so the two cannot drift apart.
    constexpr std::uint32_t DEST_ROWS_PER_TILE = 1u << DstTileSizeLog2[DstTileShape::Tile32x32];
    static_assert(DEST_ROWS_PER_TILE == TILE_NUM_FACES * DEST_ROWS_PER_FACE, "DEST slot must hold a full tile");
    static_assert(MAX_TILES_IN_HALF_DEST * DEST_ROWS_PER_TILE == DEST_REGISTER_HALF_SIZE, "DEST slots tile the bank");
    // Rows in one 16-bit DEST bank. The 32-bit bank is half this, but it is excluded below.
    constexpr std::uint32_t DEST_ROWS_PER_BANK = DEST_REGISTER_HALF_SIZE;

#pragma GCC unroll 0
    for (std::uint32_t n = 0; n < loop_count; n++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();

        // Clear DEST face-by-face when reusing dest as source
        int clear_fp32                     = is_fp32_dest_acc_en && clear_fp32_dst_acc ? 1 : 0;
        const std::uint32_t tiles_per_bank = clear_fp32 ? MAX_TILES_IN_HALF_DEST >> 1 : MAX_TILES_IN_HALF_DEST;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
        const std::uint32_t face_index     = get_dest_index_in_faces(local_tile, face_offset + n);

        // ZEROACC's 16-row mode takes an absolute 16-row block index within the DEST bank, but Blackhole
        // derives that instruction's bank-select from (dest row offset + block index) -- a row offset and a
        // block index added together. Once the sum reaches the bank size the select flips and the clear
        // lands 512 rows away in the other DEST half: the face meant to be cleared keeps the previous
        // accumulation step, and 16 unrelated rows of the other half are zeroed instead. ELWMUL accumulates
        // into DEST, so the stale face surfaces as (previous step + product); ELWADD/ELWSUB overwrite DEST
        // and issue no ZEROACC at all, which is why only ELWMUL shows it.
        //   Measured on p150 with a fixed block index of 31: dest row offset 480 -> clears block 31 (right),
        //   offset 488 or 496 -> clears block 63 (wrong half). At offset 496, indices 0..15 still land
        //   correctly and 16..31 do not -- matching the (offset + index) >= 512 boundary exactly.
        // Here the dest pointer is 64*local_tile + 16*face while the index is 4*local_tile + face, so the
        // sum only reaches 512 on the very last face of the last tile of a 16-bit bank (496 + 31 = 527).
        // A 32-bit bank tops out at 240 + 15 and never trips it. ZEROACC's one-row mode addresses purely
        // in rows (dest offset + RWC + index) and has no such mismatch, so use it for that one face.
        // tt-metal#53693.
        // The fallback is restricted to a 16-bit DEST at compile time. A 32-bit CLR_16 block is not 16
        // consecutive DEST rows, so the row-by-row clear would not be equivalent there -- and a 32-bit
        // bank cannot reach the boundary anyway (its offset tops out at 240 and its index at 15).
        // Note this keys off is_fp32_dest_acc_en, not clear_fp32: with clear_fp32_dst_acc false (the
        // LLK-level default) clear_fp32 is 0 even in FP32 mode, so it says nothing about DEST geometry.
        constexpr bool bank_is_16bit        = !is_fp32_dest_acc_en;
        const std::uint32_t dest_row_offset = (local_tile * DEST_ROWS_PER_TILE) + ((face_offset + n) * DEST_ROWS_PER_FACE);
        const bool crosses_bank             = bank_is_16bit && (dest_row_offset + face_index >= DEST_ROWS_PER_BANK);

        if (crosses_bank)
        {
            zeroacc_face_by_row();
        }
        else
        {
            TT_ZEROACC(ZERO_ACC_MODE, clear_fp32, 0, ADDR_MOD_1, face_index);
        }

        ckernel_template::run();

        if constexpr (is_high_fidelity(math_fidelity))
        {
            if (face_r_dim <= MAX_FPU_ROWS)
            {
                TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0);
            }
        }
    }
}

/**
 * @brief Run the dest-reuse op on a whole tile: one drain, every face moved into the source bank, the DEST faces
 *        cleared for the multiply, then the one-run whole-tile MOP (see @ref eltwise_binary_tile_dvalid).
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB>
 * @param num_faces: Faces of the tile, 1, 2 or 4.
 * @param clear_fp32_dst_acc: Clear the FP32 dest accumulator faces when FP32 mode is enabled.
 * @param dst_index: Tile index into the destination register.
 */
template <EltwiseBinaryType eltwise_binary_type, bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_run_with_dest_reuse_tile(const std::uint32_t num_faces, const bool clear_fp32_dst_acc, const std::uint32_t dst_index)
{
    eltwise_binary_reuse_dest_as_src_tile<binary_reuse_dest>(num_faces);

    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        // With the DEST counter at the tile base the ZEROACC bank-select sum stays below the bank size
        const int clear_fp32               = is_fp32_dest_acc_en && clear_fp32_dst_acc ? 1 : 0;
        const std::uint32_t tiles_per_bank = clear_fp32 ? MAX_TILES_IN_HALF_DEST >> 1 : MAX_TILES_IN_HALF_DEST;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            TT_ZEROACC(p_zeroacc::CLR_16, clear_fp32, 0, ADDR_MOD_1, get_dest_index_in_faces(local_tile, face));
        }
    }

    ckernel_template::run();
}

/**
 * @brief Perform elementwise binary operation with dest reuse: Output = SrcA [+, -, *] SrcB, where one src comes from the dest register.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <DEST_TO_SRCA/DEST_TO_SRCB> (NONE is rejected)
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the init
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 * @param clear_fp32_dst_acc: Clears index in destination register when float32 mode is enabled
 * @note Call @ref _llk_math_eltwise_binary_with_dest_reuse_init_ with matching template args before this function.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest,
    SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_with_dest_reuse_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc)
{
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;

    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_ for no dest reuse");
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_with_dest_reuse_", tensor_shape);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (eltwise_binary_tile_dvalid<src_b_bcast_type, binary_reuse_dest, src_dvalid>)
    {
        if (eltwise_binary_reuse_tile_shape<src_b_bcast_type>(tensor_shape))
        {
            eltwise_binary_run_with_dest_reuse_tile<eltwise_binary_type, is_fp32_dest_acc_en, binary_reuse_dest>(num_faces, clear_fp32_dst_acc, dst_index);
            math::clear_dst_reg_addr();
            return;
        }
    }

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast with dest reuse:
            // For each face row: process num_faces_c_dim faces, then CLR_B
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t face_col = 0; face_col < num_faces_c_dim; face_col++)
                {
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    ckernel_template::run();
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: process all faces sequentially
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < num_faces; n++)
            {
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run();
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast with dest reuse and multiply:
            // Process num_faces_c_dim faces per row with ZEROACC
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                // face_offset = face_row * num_faces_c_dim (position in face array)
                const std::uint32_t face_offset = face_row * num_faces_c_dim;
                eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
                    num_faces_c_dim, face_offset, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: process all faces with ZEROACC
            eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
                num_faces, 0 /*face_offset*/, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);

            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

/*************************************************************************
 * Public API - Wrapper Functions (Backward Compatible)
 *************************************************************************/

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 *
 * Dispatches to the standard or dest-reuse implementation based on the binary_reuse_dest template parameter.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the unpack init of the op (see @ref eltwise_binary_tile_dvalid)
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_init_ which feeds SrcA/SrcB.
 * @note @ref _llk_math_eltwise_binary_ runs the configured op with matching template args.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    SrcDvalid src_dvalid                         = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, src_dvalid>(tensor_shape, acc_to_dest);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest, src_dvalid>(
            tensor_shape, acc_to_dest);
    }

    // ELWADD/ELWMUL/ELWSUB read the Src zero-substitution flag but
    // eltwise-binary init never sets it, so re-establish the operand-driven DEFAULT here — otherwise a preceding
    // copy_init/datacopy op that left PRESERVE leaks into the MOP (denormal Src results differ). Also
    // covers bcast add/sub/mul, which route through this init. Mirrors reduce/transpose/datacopy.
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 *
 * Dispatches to the standard or dest-reuse implementation based on the binary_reuse_dest template parameter.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the init
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 * @param clear_fp32_dst_acc: Clears index in destination register when float32 mode is enabled
 * @note Call @ref _llk_math_eltwise_binary_init_ with matching template args before this
 *       function, and @ref _llk_math_eltwise_binary_uninit_ after it to restore modified state.
 * @note On the unpack thread, @ref _llk_unpack_AB_ must feed the operand tiles into SrcA/SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    SrcDvalid src_dvalid                         = SrcDvalid::PerFace>
inline void _llk_math_eltwise_binary_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc = false)
{
    // Zero-flag leak guard. ELWADD/ELWMUL/ELWSUB honor ALU_ACC_CTRL_Zero_Flag_disabled_src; a prior
    // copy_init/datacopy op leaking PRESERVE here changes denormal Src results. eltwise_binary_init
    // must have re-established the format-driven DEFAULT (fires only under LLK asserts).
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "eltwise_binary: Src zero-substitution flag is not in DEFAULT state — a prior op (copy_init/datacopy) leaked "
        "PRESERVE into ELWADD/ELWMUL/ELWSUB without a format-changing reconfig; denormal Src results will differ");

    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity, src_dvalid>(tensor_shape, dst_index);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_<
            eltwise_binary_type,
            src_b_bcast_type,
            Dst,
            is_fp32_dest_acc_en,
            math_fidelity,
            binary_reuse_dest,
            src_dvalid>(tensor_shape, dst_index, clear_fp32_dst_acc);
    }
}

/**
 * @brief Uninitialize/cleanup after elementwise binary operations, restoring any modified state to defaults.
 *
 * @note Reverses @ref _llk_math_eltwise_binary_init_; currently a no-op since all state is transient.
 */
inline void _llk_math_eltwise_binary_uninit_()
{
    // No state to restore - all states are transient or default
}
