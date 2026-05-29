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

using namespace ckernel;

template <BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod_custom()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 8},
        // The increment field is 6 bits wide, so 0x3F & -8 encodes a step of
        // -8 and effectively rewinds SrcB back by one face.
        .srcb = {.incr = 0x3F & -8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 24},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param num_faces: Number of faces to process (1, 2, or 4)
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type>
inline void _llk_math_eltwise_binary_init_custom_(const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod_custom<src_b_bcast_type>();

    constexpr auto bcast               = (src_b_bcast_type == BroadcastType::COL) ? p_elwise::SRCB_BCAST_COL : p_elwise::SRCB_NO_BCAST;
    const std::uint32_t num_face_pairs = num_faces / 2;

    ckernel_template tmp(num_face_pairs, 2, TT_OP_ELWSUB(p_setrwc::CLR_NONE, 0, bcast, ADDR_MOD_0, 0));
    tmp.set_loop_op1(TT_OP_ELWSUB(p_setrwc::CLR_NONE, 0, bcast, ADDR_MOD_1, 0));
    tmp.set_last_inner_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_NONE, 0, bcast, ADDR_MOD_2, 0));
    tmp.set_last_outer_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_A, 0, bcast, ADDR_MOD_2, 0));
    tmp.program();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

inline void _llk_math_sub_bcast_cols_reuse_custom_(const std::uint32_t ct_dim, const std::uint32_t dst_index)
{
    if (dst_index == 0)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

#pragma GCC unroll 0
    for (std::uint32_t tile = 0; tile < ct_dim; tile++)
    {
        ckernel_template::run();
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB);
}
