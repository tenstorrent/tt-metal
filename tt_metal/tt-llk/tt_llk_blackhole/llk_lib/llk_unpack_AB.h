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
 * @brief Configure the MOP (Micro-Operation Program) for unpacking two source operands A and B
 *
 * Sets up the unpacker MOP to handle various broadcast modes and transpose configurations.
 * The MOP programs the sequence of unpack operations based on tile geometry and broadcast type.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param transpose_of_faces: Whether to transpose faces (reorder faces 0,2,1,3)
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_mop_config_", tensor_shape);

    if (transpose_of_faces)
    {
        LLK_ASSERT(num_faces_r_dim == num_faces_c_dim, "num_faces_r_dim must be equal to num_faces_c_dim when transpose_of_faces is true");
        LLK_ASSERT(
            num_faces_c_dim == 1 || num_faces_c_dim == 2,
            "num_faces_c_dim has to be either 1 or 2 with transpose due to stride limitations in UNPACR instruction,"
            "this limitation can be removed when TensorShapes are passed at compile time");
    }

    // Transpose + Broadcast Scalar not supported
    if constexpr (BType == BroadcastType::SCALAR)
    {
        LLK_ASSERT(!transpose_of_faces, "Transpose with Broadcast Scalar not supported");
    }

    // Broadcast Row with narrow tile: only 16x16 supported, not 32x16
    if constexpr (BType == BroadcastType::ROW)
    {
        LLK_ASSERT(!(num_faces_c_dim < num_faces_r_dim), "Broadcast Row with 32x16 narrow tile not supported");
    }

    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_transpose =
        TT_OP_UNPACR(SrcA, 0b10 /*This is an inc of 2, which is meant to be num_faces_c_dim*/, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    const std::uint32_t outerloop = transpose_of_faces ? num_faces_c_dim : num_faces_r_dim;
    const std::uint32_t innerloop = transpose_of_faces ? num_faces_r_dim : num_faces_c_dim;
    const std::uint32_t srca_op   = transpose_of_faces ? unpack_srca_transpose : unpack_srca;

    // Helper to set end op(s) based on transpose mode
    auto set_end_op_with_transpose = [&](ckernel_template &tmp, std::uint32_t primary_end_op)
    {
        if (transpose_of_faces)
        {
            tmp.set_end_ops(primary_end_op, TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001));
        }
        else
        {
            tmp.set_end_op(primary_end_op);
        }
    };

    if constexpr (BType == BroadcastType::COL)
    {
        // COL broadcast: First col in Src B face is broadcast across A faces in the same row
        LLK_ASSERT(
            num_faces_c_dim >= num_faces_r_dim,
            "If num_faces_c_dim is less than num_faces_r_dim (i.e 32x16), then BROADCAST_TYPE::COL is not supported, Can be fixed in the future");
        ckernel_template tmp(outerloop, innerloop, srca_op);
        tmp.set_start_op(unpack_srcb);

        if (num_faces_c_dim < MAX_NUM_FACES_C_DIM)
        {
            set_end_op_with_transpose(
                tmp, TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 1 /*this should be num_faces_c_dim, but can't pass it here until its compile time*/, 0b0001));
        }
        else
        {
            set_end_op_with_transpose(
                tmp, TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2 /*this should be num_faces_c_dim, but can't pass it here until its compile time*/, 0b0001));
        }

        tmp.program();
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // ROW broadcast: First row in Src B face is broadcast across A faces in the same column
        LLK_ASSERT(
            num_faces_c_dim >= num_faces_r_dim,
            "If num_faces_c_dim is less than num_faces_r_dim (i.e 32x16), then BROADCAST_TYPE::ROW is not supported, Can be fixed in the future");
        static constexpr std::uint32_t unpack_srcb_clear_z = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001);
        ckernel_template tmp(outerloop, innerloop, unpack_srcb, srca_op);
        set_end_op_with_transpose(tmp, unpack_srcb_clear_z);

        tmp.program();
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // SCALAR broadcast: single B value broadcast to all A faces
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");

        ckernel_template tmp(1, tensor_shape.total_num_faces(), unpack_srca);
        tmp.set_start_op(unpack_srcb);
        tmp.program();
    }
    else // BType == BroadcastType::NONE
    {
        // NONE: no broadcast, A and B faces are paired 1:1
        if (transpose_of_faces)
        {
            static constexpr std::uint32_t srca_set_z = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001);
            // Flip r & c dimension due to transpose of SrcA, SrcA unpack increments L1 pointer by num_faces_c_dim
            ckernel_template tmp(num_faces_c_dim, num_faces_r_dim, unpack_srca_transpose, unpack_srcb);
            tmp.set_end_op(srca_set_z);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(num_faces_r_dim, num_faces_c_dim, unpack_srca, unpack_srcb);
            tmp.program();
        }
    }
}

/**
 * @brief Whether the two-operand unpack can hand each operand over as one source bank holding the whole tile: SrcDvalid::PerTile, with or
 *        without a broadcast; it does without transpose and for the shapes of @ref unpack_AB_tile_shape. The math init
 *        (@ref _llk_math_eltwise_binary_init_) applies the same rule.
 */
template <BroadcastType BType, SrcDvalid src_dvalid>
inline constexpr bool unpack_AB_tile_dvalid = src_dvalid == SrcDvalid::PerTile;

/**
 * @brief Whether a tile takes the whole-tile hand-off: full 16-row faces, and 2 x 2 faces for a column or row broadcast. Matches the math
 *        side's eltwise_binary_tile_shape.
 */
template <BroadcastType BType>
inline bool unpack_AB_tile_shape(const ckernel::TensorShape tensor_shape)
{
    constexpr bool needs_2x2_faces = BType == BroadcastType::COL || BType == BroadcastType::ROW;
    return tensor_shape.face_r_dim == FACE_R_DIM && (!needs_2x2_faces || (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 2));
}

/**
 * @brief Configure the MOP that hands each operand over as one source bank (see @ref unpack_AB_tile_dvalid): one UNPACR reads every face
 *        of A (datum count set by the init) and publishes it once; B fills its bank in the layout the math MOP reads, then publishes it.
 *        No broadcast: B's whole tile. Scalar: B's face 0. Column: faces 0, 0, 2, 2, one per 16 rows. Row: faces 0 and 1, twice.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 */
template <BroadcastType BType>
inline void _llk_unpack_AB_mop_config_tile_()
{
    static constexpr std::uint32_t unpack_srca_tile = TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_last = TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // B counters back to the tile base for the next call
    static constexpr std::uint32_t srcb_reset_zw = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);

    if constexpr (BType == BroadcastType::COL)
    {
        // AddrMode bits 5:4 step the SrcB row by one face (Ch1 Z), bits 1:0 step the L1 face (Ch0 Z)
        static constexpr std::uint32_t srcb_same_face = TT_OP_UNPACR(SrcB, 0b00010000, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        static constexpr std::uint32_t srcb_to_face_2  = TT_OP_UNPACR(SrcB, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        ckernel_template tmp(1, 2, srcb_same_face, srcb_to_face_2);
        tmp.set_start_op(unpack_srca_tile);
        tmp.set_last_inner_loop_instr(unpack_srcb_last);
        tmp.set_last_outer_loop_instr(unpack_srcb_last);
        tmp.set_end_op(srcb_reset_zw);
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        static constexpr std::uint32_t srcb_two_faces = TT_OP_UNPACR(SrcB, 0b00100000, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        ckernel_template tmp(1, 1, srcb_two_faces, unpack_srcb_last);
        tmp.set_start_op(unpack_srca_tile);
        tmp.set_end_op(srcb_reset_zw);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, 1, unpack_srca_tile, unpack_srcb_last);
        tmp.program();
    }
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 *
 * Configures the unpacker hardware for dual-operand unpacking with support for various
 * broadcast modes and optional transpose. Sets up number of datums to unpack based on face dimensions.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; must match the math init (see @ref unpack_AB_tile_dvalid)
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values = <None/IntraFace/InterFace/Both>
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 * @ref _llk_math_eltwise_binary_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_init_", tensor_shape);
    const bool within_face_16x16_transpose = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;
    const bool transpose_of_faces          = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose); // transpose within the face

    if constexpr (unpack_AB_tile_dvalid<BType, src_dvalid>)
    {
        LLK_ASSERT(
            transpose == ckernel::Transpose::None,
            "SrcDvalid::PerTile publishes per face for a transposed operand; pair a transposed unpack with SrcDvalid::PerFace on both threads");
        if (transpose == ckernel::Transpose::None && unpack_AB_tile_shape<BType>(tensor_shape))
        {
            constexpr std::uint32_t face_datums = FACE_R_DIM * FACE_C_DIM;
            const std::uint32_t tile_datums     = tensor_shape.total_num_faces() * face_datums;
            if constexpr (BType == BroadcastType::NONE)
            {
                TT_SETADCXX(p_setadc::UNP_AB, tile_datums - 1, 0x0);
            }
            else
            {
                TT_SETADCXX(p_setadc::UNP_A, tile_datums - 1, 0x0);
                TT_SETADCXX(p_setadc::UNP_B, (BType == BroadcastType::ROW ? 2 * face_datums : face_datums) - 1, 0x0);
            }
            _llk_unpack_AB_mop_config_tile_<BType>();
            return;
        }
    }

    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);

    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, tensor_shape); // transpose of faces 0,2,1,3
}

/**
 * @brief Initialize unpacker to unpack operands A and B with no transpose.
 *
 * Convenience overload that forwards to the transpose-aware init with @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile> (see the transpose-aware init)
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_AB_init_<BType, src_dvalid>(tensor_shape, ckernel::Transpose::None);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with a boolean transpose flag.
 *
 * Convenience overload taking an integer flag: nonzero selects @ref ckernel::Transpose::Both,
 * zero selects @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile> (see the transpose-aware init)
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Nonzero to enable both inter-face and within-face transpose, zero for none.
 * @note Call @ref _llk_unpack_AB_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose)
{
    _llk_unpack_AB_init_<BType, src_dvalid>(tensor_shape, transpose > 0 ? ckernel::Transpose::Both : ckernel::Transpose::None);
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
