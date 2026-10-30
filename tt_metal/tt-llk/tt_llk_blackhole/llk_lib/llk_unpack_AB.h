// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Rows per face that unpacker B reads in a two-operand unpack: only the rows the math thread consumes.
 *
 * ROW and SCALAR broadcast use SrcB row 0 alone (the math SrcB address does not advance inside a face, see
 * @ref eltwise_binary_configure_addrmod), so B reads one row per face whatever its shape. NONE and COL consume
 * every row A does, so B reads A's face rows and must have at least that many.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape_a: Tensor shape of source A
 * @param tensor_shape_b: Tensor shape of source B
 */
template <BroadcastType BType>
inline std::uint32_t _llk_unpack_AB_srcb_face_r_dim_(const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b)
{
    if constexpr (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
    {
        return 1;
    }
    else
    {
        LLK_ASSERT(tensor_shape_b.face_r_dim >= tensor_shape_a.face_r_dim, "SrcB faces must have at least as many rows as SrcA faces for NONE/COL");
        return tensor_shape_a.face_r_dim;
    }
}

/**
 * @brief Configure the MOP (Micro-Operation Program) for unpacking two source operands A and B
 *
 * SrcA walks its faces row by row (column by column with inter-face transpose). SrcB follows the broadcast:
 * - NONE: one B face per A face.
 * - COL: one B face per row of A faces, at the start of the matching row of B faces.
 * - ROW: B's first row of faces, re-read for every row of A faces.
 * - SCALAR: B's first face, read once per tile.
 *
 * @note With no broadcast and no inter-face transpose, a face shorter than MAX_FPU_ROWS rows zeroes both SrcA and
 * both SrcB banks here, because the FPU reads at least MAX_FPU_ROWS rows per face.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param transpose_of_faces: Whether to transpose faces (reorder faces 0,2,1,3)
 * @param tensor_shape_a: Tensor shape of source A; sets the face loop
 * @param tensor_shape_b: Tensor shape of source B; sets where each row of B faces starts
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b)
{
    const std::uint32_t num_faces_r_dim = tensor_shape_a.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape_a.num_faces_c_dim;
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_mop_config_", tensor_shape_a);

    if (transpose_of_faces)
    {
        LLK_ASSERT(num_faces_r_dim == num_faces_c_dim, "num_faces_r_dim must be equal to num_faces_c_dim when transpose_of_faces is true");
        LLK_ASSERT(
            num_faces_c_dim == 1 || num_faces_c_dim == 2,
            "num_faces_c_dim has to be either 1 or 2 with transpose due to stride limitations in UNPACR instruction");
    }

    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // Z increment of 2 steps down a column of faces; with transpose num_faces_c_dim is 1 or 2, and 1 has one face per column.
    static constexpr std::uint32_t unpack_srca_transpose = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    const std::uint32_t outerloop = transpose_of_faces ? num_faces_c_dim : num_faces_r_dim;
    const std::uint32_t innerloop = transpose_of_faces ? num_faces_r_dim : num_faces_c_dim;
    const std::uint32_t srca_op   = transpose_of_faces ? unpack_srca_transpose : unpack_srca;
    // With transpose, each outer pass ends at the bottom of a column of faces; move SrcA to the top of the next.
    const std::uint32_t srca_next_column = transpose_of_faces ? TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001) : TT_OP_NOP;

    if constexpr (BType == BroadcastType::COL)
    {
        LLK_ASSERT(num_faces_c_dim >= num_faces_r_dim, "BroadcastType::COL does not support a 32x16 SrcA tile");
        LLK_ASSERT(tensor_shape_b.num_faces_r_dim >= num_faces_r_dim, "BroadcastType::COL needs a SrcB face for every row of SrcA faces");
        // B's first column is broadcast across a row of A faces; then point B at the start of its next row of faces.
        ckernel_template tmp(outerloop, innerloop, srca_op);
        tmp.set_start_op(unpack_srcb);
        tmp.set_end_ops(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, tensor_shape_b.num_faces_c_dim, 0b0001), srca_next_column);
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        LLK_ASSERT(num_faces_c_dim >= num_faces_r_dim, "BroadcastType::ROW does not support a 32x16 SrcA tile");
        LLK_ASSERT(tensor_shape_b.num_faces_c_dim >= num_faces_c_dim, "BroadcastType::ROW needs a SrcB face for every column of SrcA faces");
        // B's first row of faces pairs with every row of A faces.
        ckernel_template tmp(outerloop, innerloop, unpack_srcb, srca_op);
        tmp.set_end_ops(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001), srca_next_column);
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");
        ckernel_template tmp(1, tensor_shape_a.total_num_faces(), unpack_srca);
        tmp.set_start_op(unpack_srcb);
        tmp.program();
    }
    else // BType == BroadcastType::NONE
    {
        LLK_ASSERT(tensor_shape_b.total_num_faces() >= tensor_shape_a.total_num_faces(), "SrcB needs a face for every SrcA face");
        if (!transpose_of_faces && tensor_shape_a.face_r_dim < MAX_FPU_ROWS)
        {
            // The FPU reads at least MAX_FPU_ROWS rows per face. Clear both sources so ELWADD and ELWSUB also
            // see zeros in the rows past the face.
            _llk_unpack_clear_src_banks_<SrcA>();
            _llk_unpack_clear_src_banks_<SrcB>();
        }
        ckernel_template tmp(outerloop, innerloop, srca_op, unpack_srcb);
        tmp.set_end_op(srca_next_column);
        tmp.program();
    }
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 *
 * Configures the unpacker hardware for dual-operand unpacking with support for various broadcast modes and
 * optional transpose. Each unpacker reads only what the math thread consumes from its operand: A reads its
 * full faces, and B reads the rows given by @ref _llk_unpack_AB_srcb_face_r_dim_ (one row for ROW/SCALAR).
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape_a: Tensor shape of source A (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param tensor_shape_b: Tensor shape of source B, e.g. a [1,32] tile for a ROW/SCALAR broadcast operand
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values = <None/IntraFace/InterFace/Both>
 * @note With no broadcast and no inter-face transpose, a face shorter than MAX_FPU_ROWS rows waits for math and
 * zeroes both SrcA and both SrcB banks (see @ref _llk_unpack_AB_mop_config_). Re-run this init after any op that
 * writes SrcA or SrcB, for example a MOVD2A or MOVD2B.
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 * @ref _llk_math_eltwise_binary_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_init_", tensor_shape_a);
    const bool within_face_16x16_transpose = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;
    const bool transpose_of_faces          = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose); // transpose within the face

    const std::uint32_t srcb_face_r_dim = _llk_unpack_AB_srcb_face_r_dim_<BType>(tensor_shape_a, tensor_shape_b);
    if (srcb_face_r_dim == tensor_shape_a.face_r_dim)
    {
        config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape_a.face_r_dim);
    }
    else
    {
        config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape_a.face_r_dim);
        config_unpacker_x_end<p_setadc::UNP_B>(srcb_face_r_dim);
    }

    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, tensor_shape_a, tensor_shape_b);
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B that share one tensor shape.
 *
 * Forwards to the two-shape init with B's shape equal to A's.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape of both sources (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values = <None/IntraFace/InterFace/Both>
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, tensor_shape, transpose);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with no transpose.
 *
 * Convenience overload that forwards to the transpose-aware init with @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, ckernel::Transpose::None);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with a boolean transpose flag.
 *
 * Convenience overload taking an integer flag: nonzero selects @ref ckernel::Transpose::Both,
 * zero selects @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Nonzero to enable both inter-face and within-face transpose, zero for none.
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose > 0 ? ckernel::Transpose::Both : ckernel::Transpose::None);
}

/**
 * @brief No-op teardown after AB (two-operand) unpacking.
 *
 * The SrcA/SrcB unpacker x-start/x-end (datum-count) state is transient and reprogrammed by each
 * operation's init (see tt-llk#1036), so there is nothing to restore here.
 *
 * @note Call @ref _llk_unpack_AB_init_ before this function.
 */
inline void _llk_unpack_AB_uninit_()
{
}

/**
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers
 *
 * Performs the actual unpacking operation by programming base addresses and running
 * the configured MOP. Handles context switching and synchronization with the unpacker.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param address_a: L1 memory address of source A tile
 * @param address_b: L1 memory address of source B tile
 * @param bcast_row_idx: Row index within source B tile for ROW broadcast
 * @param srcb_format: Source B data format used to calculate ROW broadcast address offset
 * @note Call @ref _llk_unpack_AB_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_uninit_ after it to restore modified state.
 * @ref _llk_math_eltwise_binary_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters

    if constexpr (BType == BroadcastType::ROW)
    {
        if (bcast_row_idx > 0)
        {
            // Row broadcast reads a full 32-element row, which spans two faces:
            //   Row 0: Face0 row 0 (cols 0-15) + Face1 row 0 (cols 16-31)
            //   Row 31: Face2 row 15 (cols 0-15) + Face3 row 15 (cols 16-31)
            //
            // Within each face, rows are stored contiguously.
            const std::uint32_t bytes_per_row_in_face = SCALE_DATUM_SIZE(srcb_format, FACE_WIDTH);
            const std::uint32_t bytes_per_face        = SCALE_DATUM_SIZE(srcb_format, FACE_WIDTH * FACE_HEIGHT);

            std::uint32_t row_offset_bytes;
            if (bcast_row_idx < FACE_HEIGHT)
            {
                // Rows 0-15 are in Face 0/1. Offset to the row within Face 0.
                row_offset_bytes = bcast_row_idx * bytes_per_row_in_face;
            }
            else
            {
                // Rows 16-31 are in Face 2/3. Skip first two faces, then offset to the row within Face 2.
                row_offset_bytes = 2 * bytes_per_face + (bcast_row_idx - FACE_HEIGHT) * bytes_per_row_in_face;
            }

            address_b += row_offset_bytes >> 4;
        }
    }

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
