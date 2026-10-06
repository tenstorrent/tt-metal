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
    return TT_OP_UNPACR_NOP(source, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1 /* wait like UNPACR */, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

// Block unpack of the plain SrcA datacopy path: _llk_unpack_A_block_ replays a recorded per tile body once per tile from one context
// acquire. The body first adds the tile stride (SCRATCH_SEC0) to the base address register of its context; half 0 is context 0, half 1 context 1.
constexpr std::uint32_t block_replay_half_len(const std::uint32_t num_faces)
{
    return 2 + 2 * num_faces; // CFGSHIFTMASK, SETADCZW, then UNPACR + SrcB dvalid NOP per face
}

// The replay record takes start and length as immediates: one instantiation per legal face count.
template <std::uint32_t base_address_reg, std::uint32_t start, std::uint32_t num_faces>
inline void load_block_replay_half()
{
    static_assert(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    load_replay_buf(
        start,
        block_replay_half_len(num_faces),
        []
        {
            // base address += SCRATCH_SEC0 (the tile stride)
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, base_address_reg);
            // ch0 Z = 0; also the one instruction between the config write and the UNPACR that consumes it
            TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0001);
            for (std::uint32_t face = 0; face < num_faces; ++face)
            {
                TTI_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
            }
        });
}

template <std::uint32_t num_faces>
inline void load_block_replay(const std::uint32_t context)
{
    if (context == 0)
    {
        load_block_replay_half<THCON_SEC0_REG3_Base_address_ADDR32, 0, num_faces>();
    }
    else
    {
        load_block_replay_half<THCON_SEC0_REG3_Base_cntx1_address_ADDR32, block_replay_half_len(num_faces), num_faces>();
    }
}

} // namespace llk_unpack_a_detail

/**
 * @brief Program the unpacker MOP for a single-operand (A) unpack.
 *
 * Selects the UNPACR instruction sequence based on broadcast type, dest-reuse mode and
 * whether data is unpacked straight to the dest register, covering transpose-of-faces and
 * 32-bit-to-dest paths.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Whether faces are reordered (transposed) during the unpack.
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
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
    const std::uint8_t num_faces = tensor_shape.total_num_faces();

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_column =
        TT_OP_UNPACR(SrcA, 0b00100010 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_transpose_of_faces =
        TT_OP_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch1_z+=1, ch0_z+=2
    static constexpr std::uint32_t unpack_srca_set_dvalid = TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_inc_z_0 =
        TT_OP_UNPACR(SrcB, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t srca_set_z_1           = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001); // set srcA ch0_z = 1
    static constexpr std::uint32_t srcb_set_z_2           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001); // set srcB ch0_z = 2
    static constexpr std::uint32_t srcb_clear_z           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001); // set srcB ch0_z = 0

    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        if (transpose_of_faces && num_faces == 4)
        {
            const std::uint32_t outerloop = 2;
            const std::uint32_t innerloop = 2;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_transpose_of_faces);
            tmp.set_end_op(TT_OP_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 1, 0b0101));
            tmp.program();
        }
        else if (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            constexpr std::uint32_t outerloop = BType == BroadcastType::ROW ? 2 : 1;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
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
        if constexpr (acc_to_dest)
        {
            // Use unpacker-bank readiness for the dummy SrcA publication so unpack can prepare
            // the next bank while math consumes the current bank.
            static constexpr std::uint32_t unpack_srca_reuse = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                                                                   ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                                                                   : unpack_srca_set_dvalid;

            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces, add support for dest to srcB
            ckernel_template tmp(outerloop, innerloop, unpack_srca_reuse, unpack_srca_reuse);
            tmp.set_start_op(unpack_srcb);
            tmp.set_end_op(srcb_set_z_2);
            tmp.program();
        }
        else
        {
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 1; // TODO: add support for num_faces
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, srcb_set_z_2);
            tmp.set_start_op(unpack_srca_set_dvalid);
            tmp.set_end_op(unpack_srcb);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        const std::uint32_t outerloop = tensor_shape.num_faces_r_dim;
        const std::uint32_t innerloop = tensor_shape.num_faces_c_dim;
        if constexpr (acc_to_dest)
        {
            // Publish one dummy SrcA DVALID per tensor face, using unpacker-bank readiness so
            // unpack can prepare the next bank while math consumes the current bank.
            static constexpr std::uint32_t unpack_srca_reuse = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                                                                   ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                                                                   : unpack_srca_set_dvalid;
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, unpack_srca_reuse);
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
                [num_faces]
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    if (num_faces > 2)
                    {
                        TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=2
                    }
                    else
                    {
                        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=1
                    }
                });

            const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
            const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, replay_buf_len)); // Unpack faces 0/2 && 1/3 to srcA
                                                                                              // or 0/1 for 2 face tile
            if (num_faces > 2)
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
 * @param transpose_of_faces: Nonzero to reorder (transpose) faces during the unpack.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose (haloize mode).
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
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
    LLK_ASSERT(BType != BroadcastType::COL || num_faces == 4, "Unary Broadcast Column requires num_faces == 4 (32x32 only)");
    LLK_ASSERT(transpose_of_faces == 0 || face_r_dim == 16, "Partial faces are not supported for transpose datacopy, face_r_dim must be 16 rows");
    LLK_ASSERT(transpose_of_faces == 0 || num_faces == 4 || num_faces == 1, "Transpose requires num_faces == 4 or 1 (32x32 and 16x16 only)");
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

    // The replay buffer holds no block body after an init: _llk_unpack_A_block_ records its own at the first call of each context.
    block_replay_body() = BlockReplayBody::None;
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
    if constexpr (((BType == BroadcastType::NONE) && (!acc_to_dest)) || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB || unpack_to_dest)
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

/**
 * @brief Unpack a block of consecutive tiles (operand A) from L1 into SrcA with one context acquire: a per tile body, recorded in the
 *        replay buffer at the first call of each context after an init, is replayed once per tile, the base address advanced by the stride in the stream.
 *
 * @tparam BType: Broadcast type, must be NONE.
 * @tparam acc_to_dest: Must be false.
 * @tparam binary_reuse_dest: Must be NONE.
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums): with a 32-bit DEST, four-face tiles take one DEST
 *         slot handshake per block; otherwise the per tile calls.
 * @tparam is_fp32_dest_acc_en: DEST holds 32-bit datums; the math thread's block call must receive the same value.
 * @param address: L1 address of the first tile of the block (16 B units).
 * @param num_tiles: Number of consecutive tiles, at least 1.
 * @param tile_stride_16B: Distance between the starts of consecutive tiles in L1 (16 B units), the operand's page size.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param num_faces: Faces per tile, the value the init received through its tensor shape.
 * @param face_r_dim: Rows per face.
 * @note Call @ref _llk_unpack_A_init_ with matching template args and transpose_of_faces 0 before this function. On the unpack to
 *       dest path the math thread takes the same block through @ref _llk_math_eltwise_unary_datacopy_block_.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false,
    bool is_fp32_dest_acc_en                     = false>
inline void _llk_unpack_A_block_(
    const std::uint32_t address,
    const std::uint32_t num_tiles,
    const std::uint32_t tile_stride_16B,
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t num_faces         = 4,
    const std::uint32_t face_r_dim        = FACE_R_DIM)
{
    static_assert(
        BType == BroadcastType::NONE && !acc_to_dest && binary_reuse_dest == EltwiseBinaryReuseDestType::NONE,
        "_llk_unpack_A_block_ supports the plain SrcA datacopy path only");
    LLK_ASSERT(num_tiles > 0, "A block has at least one tile");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address + (num_tiles - 1) * tile_stride_16B), "L1 address of the last tile must be in valid L1 memory region");

    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        // Four-face tiles lie back to back in L1 and in DEST: one DEST slot handshake for the block, the MOP once per tile, its Z
        // counters walking from tile to tile. A 16-bit DEST and other face counts take the per tile calls.
        if (!is_fp32_dest_acc_en || num_faces != 4)
        {
            for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
            {
                _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(address + tile * tile_stride_16B, unpack_src_format, unpack_dst_format);
            }
            return;
        }
        LLK_ASSERT(tile_stride_16B == face_r_dim * 16, "The unpack to dest block needs four-face 32-bit tiles back to back in L1");

        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // Clear z/w start counters
        volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
        wait_for_next_context(2);
        cfg[(unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        semaphore_post(semaphore::UNPACK_SYNC);
        set_dst_write_addr(unp_cfg_context, unpack_dst_format);
        wait_for_dest_available();
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#pragma GCC unroll 0
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            ckernel::ckernel_template::run();
        }
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
        switch_config_context(unp_cfg_context);
        return;
    }

    // Tile stride into SCRATCH_SEC0 from the instruction stream, so the write is ordered behind the CFGSHIFTMASKs of an earlier block
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_stride_16B), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_stride_16B), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);

    // Context acquire as in _llk_unpack_A_
    std::uint32_t contexts_in_use = semaphore_read(semaphore::UNPACK_SYNC);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // Clear z/w start counters
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
    std::uint32_t context                  = unp_cfg_context;
    const std::uint32_t upk0_reg           = (context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;

    // Each context replays its own half: record it at the first block call of that context after an init (every init clears the record).
    const std::uint8_t record = static_cast<std::uint8_t>(block_replay_body());
    if (__builtin_expect(record != num_faces, 0))
    {
        const std::uint8_t this_half = static_cast<std::uint8_t>(BLOCK_REPLAY_HALF_0 << context);
        if (record != (num_faces | this_half))
        {
            switch (num_faces)
            {
                case 1:
                    llk_unpack_a_detail::load_block_replay<1>(context);
                    break;
                case 2:
                    llk_unpack_a_detail::load_block_replay<2>(context);
                    break;
                default:
                    llk_unpack_a_detail::load_block_replay<4>(context);
                    break;
            }
            const bool other_half_held = record == (num_faces | (this_half ^ (BLOCK_REPLAY_HALF_0 | BLOCK_REPLAY_HALF_1)));
            block_replay_body()        = static_cast<BlockReplayBody>(other_half_held ? num_faces : (num_faces | this_half));
        }
    }

    while (contexts_in_use >= 2)
    {
        contexts_in_use = semaphore_read(semaphore::UNPACK_SYNC);
    }

    // The body adds the stride before every tile, the first one included
    cfg[upk0_reg] = address - tile_stride_16B;

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Hold the body's config write and its UNPACRs until the base address store from the RISC has landed
    TTI_STALLWAIT(p_stall::STALL_UNPACK | p_stall::STALL_CFG, p_stall::TRISC_CFG);

    const std::uint32_t half_len = llk_unpack_a_detail::block_replay_half_len(num_faces);
    const std::uint32_t start    = (context == 0) ? 0 : half_len;
    for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
    {
        TT_REPLAY(start, half_len, 0, 0);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context_from(context);
}
