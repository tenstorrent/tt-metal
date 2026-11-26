// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "llk_unpack_A.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <
    std::uint32_t num_tiles,
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_bcast_dest_reuse_mop_config_(
    const bool transpose_of_faces,
    const std::uint32_t num_faces,
    [[maybe_unused]] const std::uint32_t unpack_src_format,
    [[maybe_unused]] const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        ((BType == BroadcastType::SCALAR || BType == BroadcastType::COL) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)),
        "Not supported configuration!");
    static_assert(
        (((BType == BroadcastType::NONE) && (!acc_to_dest) && (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)) || (!unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    if (num_faces == 1)
    {
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = num_tiles;
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb_set_dvalid);
        tmp.program();
    }
    else if (transpose_of_faces)
    {
        LLK_ASSERT(num_tiles == 1, "bcast_dest_reuse transpose path supports num_tiles==1 only");
        LLK_ASSERT(num_faces == 4, "bcast_dest_reuse transpose path supports num_faces==4 only");
        constexpr std::uint32_t replay_buf_len = 5;
        load_replay_buf(
            0,
            replay_buf_len,
            []
            {
                TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // face 0 (z=0->2)
                TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // face 2 (z=2->4)
                TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001);                               // srcA ch0_z = 1
                TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // face 1 (z=1->3)
                TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // face 3 (z=3->5)
            });
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = 1;
        ckernel_template tmp(outerloop, innerloop, TT_OP_REPLAY(0, replay_buf_len, 0, 0));
        tmp.set_start_op(unpack_srcb_set_dvalid);
        tmp.program();
    }
    else
    {
        constexpr std::uint32_t outerloop = 1;
        const std::uint32_t innerloop     = num_tiles * num_faces / 2;
        ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srca);
        tmp.set_start_op(unpack_srcb_set_dvalid);
        tmp.program();
    }
}

/**
 * @brief Initialize the unpacker for the bcast dest-reuse op (SCALAR or COL): num_tiles tiles of SrcA from one
 * _llk_unpack_A_ call, SrcB fed a dummy dvalid for the MOVD2B on the math thread.
 *
 * Owns unpacker A's X span: every real UNPACR of the MOP targets SrcA, so this init programs x_end on
 * unpacker A (where DEST_TO_SRCB also routes the operand address) and does not inherit it; Tile_x_dim, the
 * Y stride and the XY counters are inherited and must still be canonical.
 *
 * @tparam num_tiles: Tiles the MOP walks per _llk_unpack_A_ call.
 * @tparam BType/acc_to_dest/binary_reuse_dest: Must be (SCALAR or COL) / true / DEST_TO_SRCB (static_assert).
 * @param transpose_of_faces, within_face_16x16_transpose: Fold a 32x32 transpose into the unpack (num_tiles == 1,
 *        num_faces == 4 only).
 * @param face_r_dim: Rows per face; sets the X span.
 * @param num_faces: Faces per tile (1, 2 or 4).
 * @note Pairs with @ref _llk_math_bcast_dest_reuse_init_ on the math thread. The matching execute is the generic
 *       @ref _llk_unpack_A_ with the same `<BType, true, DEST_TO_SRCB>` arguments; other arguments route the L1
 *       address to unpacker B, which the static_assert here cannot see.
 */
template <
    std::uint32_t num_tiles,
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_bcast_dest_reuse_init_(
    const std::uint32_t transpose_of_faces          = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // Set transpose register to prevent state pollution
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    // x-start/x-end is per-unpacker state, so program it on the unpacker the MOP issues real (non-ZEROSRC)
    // UNPACRs against. The only configurations this init accepts (SCALAR/COL + acc_to_dest + DEST_TO_SRCB, see
    // the static_assert in _llk_unpack_A_bcast_dest_reuse_mop_config_) read L1 on SrcA alone: SrcB is fed a
    // ZEROSRC UNPACR_NOP for the MOVD2B on the math thread and never reads L1, so its X counter is unused.
    // That is also where _llk_unpack_A_ routes the L1 base address for DEST_TO_SRCB whatever the broadcast type
    // (_llk_unpack_A_address_on_unpacker_A_), so the X span and the address must land on the same unpacker.
    static_assert(
        _llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(),
        "bcast_dest_reuse unpack: the X span is programmed on unpacker A, so the operand address must be routed there too");
    config_unpacker_x_end<p_setadc::UNP_A>(face_r_dim);

    _llk_unpack_A_bcast_dest_reuse_mop_config_<num_tiles, BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}
