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
 * @param tile_dvalid: One UNPACR and one data valid per operand per tile instead of per face (see @ref unpack_AB_tile_dvalid)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape, const bool tile_dvalid = false)
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
        else if (tile_dvalid)
        {
            // One UNPACR per operand moves every face of the tile (the datum count programmed by the init) into one
            // source bank, rows 0 to 16 x num_faces - 1, and publishes it once; the L1 face counter stays at 0
            // because the whole tile is one contiguous read from the base address.
            static constexpr std::uint32_t unpack_srca_tile = TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            static constexpr std::uint32_t unpack_srcb_tile = TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            ckernel_template tmp(1, 1, unpack_srca_tile, unpack_srcb_tile);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(num_faces_r_dim, num_faces_c_dim, unpack_srca, unpack_srcb);
            tmp.program();
        }
    }
}

// RISC-side mirror of the unpacker registers that the SrcDvalid::PerTile form of _llk_unpack_AB_ programs through the
// instruction stream: the base address register of each unpacker in each config context (16-byte words) and the two
// strides in SCRATCH_SEC0 (unpacker 0, operand A) and SCRATCH_SEC1 (unpacker 1, operand B) that its CFGSHIFTMASK steps
// add. UNPACK_AB_UNKNOWN marks a base register another operation may have written and a stride of 0 an unknown scratch
// register (a call never steps by 0: an unchanged address issues nothing): the PerTile init forgets every entry and
// the first call after it writes the registers in full.
constexpr std::uint32_t UNPACK_AB_UNKNOWN = 0xFFFFFFFF;
static std::uint32_t unpack_AB_base[2][2] = {{UNPACK_AB_UNKNOWN, UNPACK_AB_UNKNOWN}, {UNPACK_AB_UNKNOWN, UNPACK_AB_UNKNOWN}}; // [context][unpacker]
static std::uint32_t unpack_AB_stride[2]  = {0, 0};                                                                        // [unpacker]

/**
 * @brief Forget the unpacker registers the PerTile form of @ref _llk_unpack_AB_ tracks.
 */
inline void unpack_AB_forget_registers()
{
    for (std::uint32_t c = 0; c < 2; c++)
    {
        unpack_AB_base[c][0] = UNPACK_AB_UNKNOWN;
        unpack_AB_base[c][1] = UNPACK_AB_UNKNOWN;
        unpack_AB_stride[c]  = 0;
    }
}

/**
 * @brief How a base address register gets from the address it holds to the address a call needs.
 *
 * @param tracked: The mirror of the register.
 * @param needed: The address the call needs.
 * @param stride: The mirror of the scratch register of that unpacker.
 * @return 0 nothing to do, 1 one CFGSHIFTMASK step by the stride in the scratch register, 2 a full write, 3 a full
 *     write that also loads the scratch register with the new stride (the distance to the previous address, so a
 *     loop over consecutive tiles takes the step from its second call on).
 */
inline std::uint32_t unpack_AB_base_mode(const std::uint32_t tracked, const std::uint32_t needed, const std::uint32_t stride)
{
    if (tracked == needed)
    {
        return 0;
    }
    if (tracked == UNPACK_AB_UNKNOWN)
    {
        return 2;
    }
    return ((needed - tracked) == stride) ? 1 : 3;
}

/**
 * @brief Issue the config instructions that move the two base address registers of one config context.
 *
 * Mode 1 adds the scratch register to the base register (CFGSHIFTMASK operation 0b011 with a 32-bit mask; scratch_sel
 * 0 is SCRATCH_SEC0, 1 is SCRATCH_SEC1). Modes 2 and 3 write the register from the GPR the caller loaded, mode 3 also
 * the scratch register. The caller issues the STALLWAIT that lets the GPR writes land before the WRCFGs.
 *
 * @tparam REG_A: Base address register of unpacker 0 in the context.
 * @tparam REG_B: Base address register of unpacker 1 in the context.
 * @param mode_a: Mode of operand A (see @ref unpack_AB_base_mode).
 * @param mode_b: Mode of operand B.
 */
template <std::uint32_t REG_A, std::uint32_t REG_B>
inline void unpack_AB_apply_base(const std::uint32_t mode_a, const std::uint32_t mode_b)
{
    if (mode_a == 1)
    {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, REG_A);
    }
    else if (mode_a >= 2)
    {
        TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, REG_A);
        if (mode_a == 3)
        {
            TTI_WRCFG(p_gpr_unpack::UNPACK_AB_STRIDE_A, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        }
    }
    if (mode_b == 1)
    {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 1, REG_B);
    }
    else if (mode_b >= 2)
    {
        TTI_WRCFG(p_gpr_unpack::TMP1, p_cfg::WRCFG_32b, REG_B);
        if (mode_b == 3)
        {
            TTI_WRCFG(p_gpr_unpack::UNPACK_AB_STRIDE_B, p_cfg::WRCFG_32b, SCRATCH_SEC1_val_ADDR32);
        }
    }
}

/**
 * @brief Whether the two-operand unpack hands each operand over as one source bank holding the whole tile.
 *
 * True for @ref SrcDvalid::PerTile without broadcast, without transpose and with full 16-row faces. Every other
 * combination keeps the per-face program: the broadcast forms read SrcB per face, a transposed operand is reordered
 * face by face, and a partial face takes its 16-row spacing in the source register from one UNPACR per face. The
 * math init (@ref _llk_math_eltwise_binary_init_) applies the same rule to the same tensor shape, so the two threads
 * agree whenever they are given the same SrcDvalid; a transposed operand must be paired with SrcDvalid::PerFace on
 * both sides.
 */
template <BroadcastType BType, SrcDvalid src_dvalid>
inline constexpr bool unpack_AB_tile_dvalid(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    return src_dvalid == SrcDvalid::PerTile && BType == BroadcastType::NONE && transpose == ckernel::Transpose::None &&
           tensor_shape.face_r_dim == FACE_R_DIM;
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 *
 * Configures the unpacker hardware for dual-operand unpacking with support for various
 * broadcast modes and optional transpose. Sets up number of datums to unpack based on face dimensions.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; PerTile unpacks each operand tile with one
 *     UNPACR and publishes it once (see @ref unpack_AB_tile_dvalid for when it applies) and must be paired with the
 *     same value on the math init
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
    LLK_ASSERT(
        src_dvalid == SrcDvalid::PerFace || transpose == ckernel::Transpose::None,
        "SrcDvalid::PerTile publishes per face for a transposed operand; pair a transposed unpack with SrcDvalid::PerFace on both threads");
    const bool within_face_16x16_transpose = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;
    const bool transpose_of_faces          = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose); // transpose within the face

    const bool tile_dvalid = unpack_AB_tile_dvalid<BType, src_dvalid>(tensor_shape, transpose);
    if (tile_dvalid)
    {
        // Both unpackers read every face of the tile with one UNPACR: the datum count is the whole tile.
        const std::uint32_t x_end = tensor_shape.total_num_faces() * FACE_R_DIM * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_AB, x_end, 0x0);
    }
    else
    {
        config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);
    }

    if constexpr (src_dvalid == SrcDvalid::PerTile)
    {
        // Another operation may have written the base address and scratch registers since the last PerTile call; the
        // first call after this init writes them in full.
        unpack_AB_forget_registers();
    }

    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, tensor_shape, tile_dvalid); // transpose of faces 0,2,1,3
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
 * Programs the two base addresses and runs the configured MOP. With SrcDvalid::PerFace (the default) the addresses
 * are written from the RISC into the config context the call runs in, once the context semaphore says the UNPACRs
 * that last used that context have been accepted. With SrcDvalid::PerTile they are written through the instruction
 * stream instead: one CFGSHIFTMASK step when the address is one learned stride past the register (a loop over the
 * tiles of a circular buffer from its second call on), a SETDMAREG and WRCFG write otherwise. The thread orders those
 * writes after the UNPACRs of the previous call by itself, so the PerTile call makes no RISC register write and no
 * semaphore read, which on Blackhole were the per-call cost that held a two-operand tile at 27 cycles against the
 * 16 cycles its data takes. Both forms post the UNPACK_SYNC token from the RISC before their UNPACRs and take it back
 * in the instruction stream after them, so the RISC-side pollers of the other unpack operations still see how many
 * calls the unpacker has yet to accept, and both switch the config context at the end.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam src_dvalid: Source bank hand-off, values = <PerFace/PerTile>; the same value as the init's
 * @param address_a: L1 memory address of source A tile
 * @param address_b: L1 memory address of source B tile
 * @param bcast_row_idx: Row index within source B tile for ROW broadcast
 * @param srcb_format: Source B data format used to calculate ROW broadcast address offset
 * @note Call @ref _llk_unpack_AB_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_uninit_ after it to restore modified state.
 * @ref _llk_math_eltwise_binary_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */

template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
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

    if constexpr (src_dvalid == SrcDvalid::PerTile)
    {
        LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
        LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

        // Base addresses through the instruction stream (see the function description).
        const std::uint32_t context = unp_cfg_context;
        std::uint32_t &tracked_a    = unpack_AB_base[context][0];
        std::uint32_t &tracked_b    = unpack_AB_base[context][1];
        const std::uint32_t mode_a  = unpack_AB_base_mode(tracked_a, address_a, unpack_AB_stride[0]);
        const std::uint32_t mode_b  = unpack_AB_base_mode(tracked_b, address_b, unpack_AB_stride[1]);

        if (mode_a >= 2)
        {
            TT_SETDMAREG(0, LOWER_HALFWORD(address_a), 0, LO_16(p_gpr_unpack::TMP0));
            TT_SETDMAREG(0, UPPER_HALFWORD(address_a), 0, HI_16(p_gpr_unpack::TMP0));
            if (mode_a == 3)
            {
                const std::uint32_t stride = address_a - tracked_a;
                TT_SETDMAREG(0, LOWER_HALFWORD(stride), 0, LO_16(p_gpr_unpack::UNPACK_AB_STRIDE_A));
                TT_SETDMAREG(0, UPPER_HALFWORD(stride), 0, HI_16(p_gpr_unpack::UNPACK_AB_STRIDE_A));
                unpack_AB_stride[0] = stride;
            }
        }
        if (mode_b >= 2)
        {
            TT_SETDMAREG(0, LOWER_HALFWORD(address_b), 0, LO_16(p_gpr_unpack::TMP1));
            TT_SETDMAREG(0, UPPER_HALFWORD(address_b), 0, HI_16(p_gpr_unpack::TMP1));
            if (mode_b == 3)
            {
                const std::uint32_t stride = address_b - tracked_b;
                TT_SETDMAREG(0, LOWER_HALFWORD(stride), 0, LO_16(p_gpr_unpack::UNPACK_AB_STRIDE_B));
                TT_SETDMAREG(0, UPPER_HALFWORD(stride), 0, HI_16(p_gpr_unpack::UNPACK_AB_STRIDE_B));
                unpack_AB_stride[1] = stride;
            }
        }
        if (mode_a >= 2 || mode_b >= 2)
        {
            // The GPR writes run on THCON; the WRCFGs read the GPRs.
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        }
        if (context == 0)
        {
            unpack_AB_apply_base<THCON_SEC0_REG3_Base_address_ADDR32, THCON_SEC1_REG3_Base_address_ADDR32>(mode_a, mode_b);
        }
        else
        {
            unpack_AB_apply_base<THCON_SEC0_REG3_Base_cntx1_address_ADDR32, THCON_SEC1_REG3_Base_cntx1_address_ADDR32>(mode_a, mode_b);
        }
        if (mode_a != 0 || mode_b != 0)
        {
            // A config write takes two cycles; the UNPACRs must see the new base addresses.
            TTI_NOP;
        }
        tracked_a = address_a;
        tracked_b = address_b;
    }
    else
    {
        // Program srcA and srcB base addresses
        volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

        // Wait for free context
        wait_for_next_context(2);

        // Validate and configure addresses
        _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    }

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
