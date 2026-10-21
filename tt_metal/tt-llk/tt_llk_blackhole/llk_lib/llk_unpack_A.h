// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace llk_unpack_a_detail
{
template <EltwiseBinaryReuseDestType binary_reuse_dest>
constexpr std::uint32_t dest_reuse_dummy_unpack()
{
    static_assert(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB);
    constexpr std::uint32_t source = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? SrcA : SrcB;
    return TT_OP_UNPACR_NOP(source, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

} // namespace llk_unpack_a_detail

/**
 * @brief Whether @ref _llk_unpack_A_ programs the operand's L1 address into unpacker A (SEC0).
 *
 * Otherwise the address goes to unpacker B (SEC1): a broadcast or acc_to_dest operand's address goes
 * to unpacker B unless DEST is reused as SrcB or the operand is unpacked to DEST. For example,
 * acc_to_dest with DEST_TO_SRCA unpacks the operand to SrcB and moves DEST to SrcA.
 * This only selects the address register; the MOP decides which sources are actually unpacked.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @return true if the operand's L1 base address goes to unpacker A (SEC0), false if it goes to unpacker B (SEC1).
 */
template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
constexpr bool _llk_unpack_A_address_on_unpacker_A_()
{
    return ((BType == BroadcastType::NONE) && (!acc_to_dest)) || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB || unpack_to_dest;
}

/**
 * @brief Whether the unpack A configuration asserts check unpacker B (SEC1) rather than unpacker A (SEC0).
 *
 * Unpacker B is checked only when it both holds the operand's address (see
 * @ref _llk_unpack_A_address_on_unpacker_A_) and unpacks it from L1: broadcasts, and acc_to_dest with
 * DEST_TO_SRCA. acc_to_dest without DEST reuse also routes the address to unpacker B, but the
 * transpose-of-faces MOP that transpose_init configures with it unpacks only SrcA (SrcB is zeroed),
 * and its paired per-tile unpack uses acc_to_dest = false, so unpacker A is checked there.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @return true if the asserts should check unpacker B (SEC1), false for unpacker A (SEC0).
 */
template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
constexpr bool _llk_unpack_A_asserts_check_unpacker_B_()
{
    return !_llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>() &&
           ((BType != BroadcastType::NONE) || (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA));
}

/**
 * @brief Program the unpacker MOP for a single-operand (A) unpack.
 *
 * Selects the UNPACR instruction sequence based on broadcast type, dest-reuse mode and
 * whether data is unpacked straight to the dest register, covering transpose-of-faces and
 * 32-bit-to-dest paths. The MOP loops follow the tile's face grid (tt-metal#47328), except where
 * noted below.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Whether to transpose the face grid during the unpack. Only a 2x2 grid
 *        (32x32 tile) changes the face order; 1x1, 1x2 and 2x1 grids keep the L1 face order.
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note Unsupported on tiles other than 32x32 (asserted in @ref _llk_unpack_A_init_): COL broadcast
 *       with acc_to_dest or unpack to dest. Transpose of faces works with unpack to dest on every
 *       tile shape.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_mop_config_(
    const bool transpose_of_faces, const ckernel::TensorShape tensor_shape, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        !((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");
    static_assert(
        !(((acc_to_dest) || (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)) && (unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_A_mop_config_", tensor_shape);
    const std::uint8_t num_faces       = tensor_shape.total_num_faces();
    const std::uint8_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint8_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    // Transpose of faces reads input face (j, i) for output face (i, j). Only a 2x2 grid changes the
    // order; 1x1, 1x2 and 2x1 keep the L1 face order (1x2 becomes 2x1 and the reverse).
    const bool reorder_faces = transpose_of_faces && (num_faces_r_dim == MAX_NUM_FACES_R_DIM) && (num_faces_c_dim == MAX_NUM_FACES_C_DIM);

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_column =
        TT_OP_UNPACR(SrcA, 0b00100010 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_transpose_of_faces =
        TT_OP_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch1_z+=1, ch0_z+=2
    static constexpr std::uint32_t unpack_srca_set_dvalid =
        TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_inc_z_0 =
        TT_OP_UNPACR(SrcB, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_inc_z_2 =
        TT_OP_UNPACR(SrcB, 0b10 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid =
        TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t srca_set_z_1 = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001); // set srcA ch0_z = 1
    static constexpr std::uint32_t srcb_clear_z = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001); // set srcB ch0_z = 0

    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        if (reorder_faces)
        {
            const std::uint32_t outerloop = 2;
            const std::uint32_t innerloop = 2;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_transpose_of_faces);
            tmp.set_end_op(TT_OP_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 1, 0b0101));
            tmp.program();
        }
        else if (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            // ROW: row 0 of the top faces goes to row 0 of dest faces 0 and 1; math broadcasts them.
            // A tile with one face column reads face 0 for both, so the source face does not step.
            // SCALAR unpacks one face (outerloop = 1): row 0 of face 0 to row 0 of dest face 0.
            static constexpr std::uint32_t unpack_srca_to_dest_same_face =
                TT_OP_UNPACR(SrcA, 0b00010000 /*CH1 Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            constexpr std::uint32_t outerloop = BType == BroadcastType::ROW ? 2 : 1;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, (num_faces_c_dim == 1) ? unpack_srca_to_dest_same_face : unpack_srca_to_dest);
            tmp.program();
        }
        else if (BType == BroadcastType::COL)
        {
            constexpr std::uint32_t outerloop = 2;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_column);
            tmp.program();
        }
        else
        {
            const std::uint32_t outerloop     = num_faces;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        // One SrcB face per face row: the left face of each face row holds the column to broadcast.
        // Faces are stored row-major, so step the SrcB face counter by num_faces_c_dim.
        const std::uint32_t unpack_srcb_next_face_row = (num_faces_c_dim == MAX_NUM_FACES_C_DIM) ? unpack_srcb_inc_z_2 : unpack_srcb;
        if constexpr (acc_to_dest)
        {
            // Per face row: one SrcB face, then one dummy SrcA DVALID per face in the row.
            // The dummy SrcA publication is the same word whether or not dest is reused as SrcA.
            // TODO: add support for dest to srcB (tt-metal#47328)
            const std::uint32_t outerloop = num_faces_r_dim;
            const std::uint32_t innerloop = num_faces_c_dim;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_set_dvalid);
            tmp.set_start_op(unpack_srcb_next_face_row);
            tmp.program();
        }
        else
        {
            const std::uint32_t outerloop = 1;
            const std::uint32_t innerloop = num_faces_r_dim;
            ckernel_template tmp(outerloop, innerloop, unpack_srcb_next_face_row);
            tmp.set_start_op(unpack_srca_set_dvalid);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // One SrcB face per tile face: every face in face column c takes row 0 of face (0, c),
        // so unpack the first face row once per face row.
        const std::uint32_t outerloop = num_faces_r_dim;
        const std::uint32_t innerloop = num_faces_c_dim;
        if constexpr (acc_to_dest)
        {
            // Publish one dummy SrcA DVALID per tensor face; the word is the same whether or not
            // dest is reused as SrcA.
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, unpack_srca_set_dvalid);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = 1;
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_inc_z_0);
        tmp.set_start_op(unpack_srca_set_dvalid);
        tmp.program();
    }
    else
    {
        if (transpose_of_faces)
        {
            constexpr std::uint32_t replay_buf_len = 2;
            load_replay_buf(
                0,
                replay_buf_len,
                [reorder_faces]
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    if (reorder_faces)
                    {
                        TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=2
                    }
                    else
                    {
                        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=1
                    }
                });

            // 2x2: unpack faces 0/2, then 1/3. Other grids: unpack the faces in L1 order.
            const std::uint32_t outerloop = reorder_faces ? num_faces_c_dim : 1;
            const std::uint32_t innerloop = reorder_faces ? num_faces_r_dim : num_faces;
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, replay_buf_len));
            if (reorder_faces)
            {
                tmp.set_end_op(srca_set_z_1);
            }
            tmp.program();
        }
        else
        {
            if constexpr (acc_to_dest)
            {
                // Use unpacker-bank readiness for destination-reuse dummy source publication so
                // unpack can prepare the next bank while math consumes the current bank.
                static constexpr std::uint32_t unpack_srca_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                        : unpack_srca;

                static constexpr std::uint32_t unpack_srcb_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCB>()
                        : unpack_srcb;

                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srca_reuse, unpack_srcb_reuse);
                tmp.program();
            }
            else
            {
                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srcb_set_dvalid);
                tmp.set_start_op(unpack_srca);
                tmp.program();
            }
        }
    }
}

/**
 * @brief Initialize the unpacker for a single-operand (A) unpack.
 *
 * Configures the within-face transpose register and per-unpacker datum count, then programs
 * the MOP for the requested broadcast/dest-reuse/unpack-to-dest mode.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Nonzero to transpose the face grid during the unpack. Only a 2x2 grid
 *        (32x32 tile) changes the face order; 1x1, 1x2 and 2x1 grids keep the L1 face order.
 *        Needs 16-row faces.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose (haloize mode).
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note Asserted unsupported on tiles other than 32x32 (tt-metal#47328): COL broadcast with
 *       acc_to_dest or unpack to dest.
 * @note Asserted unsupported on every tile shape, 32x32 included (tt-llk#1702): within-face
 *       transpose with unpack to dest. Use transpose_of_faces here and transpose within faces on
 *       the math thread (llk_math_transpose_dest). Transpose of faces alone works with unpack to
 *       dest on every tile shape.
 * @note Call @ref _llk_unpack_A_uninit_ as the matching teardown; it is currently a no-op
 *       because unpacker X counters are reprogrammed by init.
 * @ref _llk_unpack_A_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ is the matching init on the math thread (datacopy/transpose consumer).
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_init_(
    const std::uint32_t transpose_of_faces          = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const ckernel::TensorShape tensor_shape         = ckernel::DEFAULT_TENSOR_SHAPE,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_A_init_", tensor_shape);
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    // Column broadcast through SrcB follows the face grid for every tile shape. Not on other
    // tiles yet (tt-metal#47328): the unpack-to-dest (32-bit) MOP always unpacks two faces in the
    // 32x32 layout, and the acc_to_dest MOP follows the face grid but has no test on them.
    LLK_ASSERT(
        BType != BroadcastType::COL || num_faces == MAX_NUM_FACES ||
            (!acc_to_dest && !should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format)),
        "Unary Broadcast Column: tiles other than 32x32 need no acc_to_dest and no unpack to dest");
    // Row broadcast with acc_to_dest follows the face grid on every tile shape; 16x32 is covered by
    // TensixBinaryComputeTinyTileRowBroadcastWithDestReuse. (Scalar broadcast with acc_to_dest is
    // rejected by a static_assert in _llk_unpack_A_mop_config_.)
    LLK_ASSERT(transpose_of_faces == 0 || face_r_dim == 16, "Partial faces are not supported for transpose datacopy, face_r_dim must be 16 rows");
    // The unpacker's within-face transpose (haloize mode) gives wrong results with unpack to dest
    // on every tile shape, 32x32 included (tenstorrent/tt-llk#1702 item 15). The compute API
    // transposes within faces on the math thread instead (transpose_dest).
    LLK_ASSERT(
        within_face_16x16_transpose == 0 || !should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format),
        "Within-face transpose is not supported with unpack to dest; use transpose_dest on the math thread (tt-llk#1702)");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

    // Set transpose register to prevent state pollution
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    // x-start/x-end is per-unpacker state, so program it on exactly the unpacker(s) the MOP issues a
    // real (non-ZEROSRC) UNPACR against; a zeroed source does not read L1, so its X counter is unused.
    // The unpack-to-dest (SrcA) path is only taken when the input is actually 32-bit; otherwise the MOP
    // falls through to the normal/broadcast path, so gate on the shared should_unpack_to_dest() predicate
    // the MOP uses, so the two cannot diverge.
    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        // SrcA -> dest. ROW and SCALAR broadcast only unpack a single row; everything else a full face.
        if constexpr (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            config_unpacker_x_end<p_setadc::UNP_A>(1);
        }
        else
        {
            config_unpacker_x_end<p_setadc::UNP_A>(face_r_dim);
        }
    }
    else
    {
        //   plain datacopy            -> SrcA           (unpacker A)
        //   acc_to_dest, no reuse      -> SrcA and SrcB  (both)
        //   acc_to_dest DEST_TO_SRCA   -> SrcB only      (SrcA comes from DEST, e.g. hardswish x*hardsigmoid(x))
        //   acc_to_dest DEST_TO_SRCB   -> SrcA only      (SrcB comes from DEST)
        //   broadcast                  -> SrcB
        constexpr bool reads_srca       = (BType == BroadcastType::NONE) && !(acc_to_dest && binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA);
        constexpr bool reads_srcb       = (BType != BroadcastType::NONE) || (acc_to_dest && binary_reuse_dest != EltwiseBinaryReuseDestType::DEST_TO_SRCB);
        constexpr std::uint32_t UNP_SEL = (reads_srca && reads_srcb) ? p_setadc::UNP_AB : (reads_srca ? p_setadc::UNP_A : p_setadc::UNP_B);
        config_unpacker_x_end<UNP_SEL>(face_r_dim);
    }

    _llk_unpack_A_mop_config_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, tensor_shape, unpack_src_format, unpack_dst_format);
}

/**
 * @brief No-op teardown after single-operand (A) unpacking.
 *
 * The unpacker x-start/x-end (datum-count) state is transient and reprogrammed by each operation's
 * init (see tt-llk#1036), so there is nothing to restore here.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @note Call @ref _llk_unpack_A_init_ with matching template args before this function.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_A_uninit_()
{
}

/**
 * @brief Unpack a single tile (operand A) from L1 into the SrcA/SrcB or dest register.
 *
 * Programs the operand base address into the active config context, synchronizes with the
 * unpacker via semaphores, and runs the configured MOP. When unpacking 32-bit datums to dest,
 * also manages the dest write address and completion handshake.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param address: L1 address of the source tile.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note Call @ref _llk_unpack_A_init_ with matching template args before this function, and
 *       @ref _llk_unpack_A_uninit_ after it as the matching teardown (currently a no-op).
 * @ref _llk_math_eltwise_unary_datacopy_ on the math thread consumes the tile unpacked here.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_(const std::uint32_t address, const std::uint32_t unpack_src_format = 0, const std::uint32_t unpack_dst_format = 0)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Set upk0/1 L1 read addr
    if constexpr (_llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>())
    {
        const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
        cfg[upk0_reg]                = address;
    }
    else
    {
        const std::uint32_t upk1_reg = (unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;
        cfg[upk1_reg]                = address;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    if constexpr (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            wait_for_dest_available();
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
        }
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
