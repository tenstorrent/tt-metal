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
#include "sanitizer/api.h"

using namespace ckernel;
using namespace ckernel::unpacker;

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
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_mop_config_(
    const bool transpose_of_faces, const std::uint32_t num_faces, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        !((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");
    static_assert(
        !(((acc_to_dest) || (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)) && (unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

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
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces, add support for dest to srcB
            ckernel_template tmp(outerloop, innerloop, unpack_srca_set_dvalid, unpack_srca_set_dvalid);
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
        constexpr std::uint32_t innerloop = 2;
        constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces
        if constexpr (acc_to_dest)
        {
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
                static constexpr std::uint32_t unpack_srca_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? unpack_srca_set_dvalid : unpack_srca;

                static constexpr std::uint32_t unpack_srcb_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? unpack_srcb_set_dvalid : unpack_srcb;

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
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note Call @ref _llk_unpack_A_uninit_ to restore the modified datum-count state.
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
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(BType != BroadcastType::COL || num_faces == 4, "Unary Broadcast Column requires num_faces == 4 (32x32 only)");
    LLK_ASSERT(transpose_of_faces == 0 || face_r_dim == 16, "Partial faces are not supported for transpose datacopy, face_r_dim must be 16 rows");
    LLK_ASSERT(transpose_of_faces == 0 || num_faces == 4 || num_faces == 1, "Transpose requires num_faces == 4 or 1 (32x32 and 16x16 only)");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

    if constexpr (BType == BroadcastType::NONE)
    {
        llk::san::unpack_operand_check(
            llk::san::IGNORE,
            unpack_src_format,
            llk::san::IGNORE,
            unpack_dst_format,
            llk::san::IGNORE,
            face_r_dim,
            llk::san::IGNORE,
            num_faces,
            llk::san::IGNORE);
    }
    else
    {
        // If using broadcast UnpackA uses UNP_B... yeah i know...
        llk::san::unpack_operand_check(
            llk::san::IGNORE,
            llk::san::IGNORE,
            unpack_src_format,
            llk::san::IGNORE,
            unpack_dst_format,
            llk::san::IGNORE,
            face_r_dim,
            llk::san::IGNORE,
            num_faces);
    }
    llk::san::operation_init<llk::san::Operation::UnpackA>(BType, acc_to_dest, binary_reuse_dest, unpack_to_dest, unpack_src_format, unpack_dst_format);

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
        constexpr bool reads_srca =
            (BType == BroadcastType::NONE) && !(acc_to_dest && binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA);
        constexpr bool reads_srcb =
            (BType != BroadcastType::NONE) || (acc_to_dest && binary_reuse_dest != EltwiseBinaryReuseDestType::DEST_TO_SRCB);
        constexpr std::uint32_t UNP_SEL = (reads_srca && reads_srcb) ? p_setadc::UNP_AB : (reads_srca ? p_setadc::UNP_A : p_setadc::UNP_B);
        config_unpacker_x_end<UNP_SEL>(face_r_dim);
    }

    // TODO NC: Move to TRISC1 tt-metal#36411
    if constexpr (BType != BroadcastType::NONE && unpack_to_dest)
    {
        _llk_unpack_dbg_feature_disable_();
    }
    _llk_unpack_A_mop_config_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}

/**
 * @brief Restore unpacker datum-count state after single-operand (A) unpacking.
 *
 * Resets the X-dimension address counter for the unpacker used by this broadcast mode back to
 * a full face worth of datums.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @param face_r_dim: Number of rows per face, used to compute the restored datum count.
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
 *       @ref _llk_unpack_A_uninit_ after it to restore modified state.
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

    if constexpr (BType == BroadcastType::NONE)
    {
        llk::san::unpack_operand_check(
            llk::san::IGNORE,
            unpack_src_format,
            llk::san::IGNORE,
            unpack_dst_format,
            llk::san::IGNORE,
            llk::san::IGNORE,
            llk::san::IGNORE,
            llk::san::IGNORE,
            llk::san::IGNORE);
    }
    else
    {
        // If using broadcast UnpackA uses UNP_B... yeah i know...
        llk::san::unpack_operand_check(
            llk::san::IGNORE,
            llk::san::IGNORE,
            unpack_src_format,
            llk::san::IGNORE,
            unpack_dst_format,
            llk::san::IGNORE,
            llk::san::IGNORE,
            llk::san::IGNORE,
            llk::san::IGNORE);
    }
    llk::san::operation_check<llk::san::Operation::UnpackA>(BType, acc_to_dest, binary_reuse_dest, unpack_to_dest, unpack_src_format, unpack_dst_format);

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
            unpack_to_dest_tile_done(unp_cfg_context);
        }
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
