// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
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
        .set(ADDR_MOD_7);
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
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod_custom<src_b_bcast_type>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

// CANONICAL mechanism note (other files refer here). Blocked bcast-col scaffold shared by the SDPA SUB
// path and the indexer_score MUL path; they differ only in the FPU op (selected at compile time), so
// addr-mod setup, srcB face-reuse addressing, and per-tile counter resets are single-sourced here. The
// name tracks the shared mechanism (srcB face reuse), matching the WH _llk_math_sub_bcast_cols_reuse_custom_.
// MUL path = dest-MAC head reduction: ELWMUL's dest_accum_en opcode field is 0 (MAC, not overwrite),
// and dest is cleared only by tile_regs_acquire's ZEROACC. So the caller does ONE acquire per ct_dim-
// column batch and calls once per head into the SAME dest -> dest[col] = sum_h qk[col,h]*w[h], one pack
// per column instead of per (column, head). cb_qk must be head-major.
template <EltwiseBinaryType eltwise_binary_type>
inline void _llk_math_bcast_cols_reuse_custom_(const std::uint32_t ct_dim = 1)
{
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWMUL || eltwise_binary_type == EltwiseBinaryType::ELWSUB,
        "blocked bcast-col reuse scaffold supports ELWMUL and ELWSUB only");

    // One FPU bcast-col op (mul or sub) advancing srcA/srcB/dest per `addr_mod`; op fixed at compile
    // time -> branch resolves to a single emitted instruction.
#define BCAST_COLS_OP(addr_mod)                                                       \
    do                                                                                \
    {                                                                                 \
        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)               \
        {                                                                             \
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, addr_mod, 0); \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, addr_mod, 0); \
        }                                                                             \
    } while (0)

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 24},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0x3F & -8}, // decrement srcB by 8
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_5);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB); // reset both src counters to 0

    // Per tile: walk srcA/dest across 4 faces while srcB reuses one face per pair (F0/F1 -> srcB face
    // 0, F2/F3 -> srcB face 2). Comments are dest/srcA counter values. ADDR_MOD_5 rewinds srcB -8 so
    // the pair's 2nd op rereads the same srcB face; ADDR_MOD_6 then jumps srcB +24 to the next pair.
    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        // face F0 (srcB face 0)
        BCAST_COLS_OP(ADDR_MOD_7); // 0 -> 8
        BCAST_COLS_OP(ADDR_MOD_5); // 8 -> 0

        // face F1 (srcB face 0, reused)
        BCAST_COLS_OP(ADDR_MOD_7); // 0 -> 8
        BCAST_COLS_OP(ADDR_MOD_6); // 8 -> 32

        // face F2 (srcB face 2)
        BCAST_COLS_OP(ADDR_MOD_7); // 32 -> 40
        BCAST_COLS_OP(ADDR_MOD_5); // 40 -> 32

        // face F3 (srcB face 2, reused)
        BCAST_COLS_OP(ADDR_MOD_7); // 32 -> 40
        BCAST_COLS_OP(ADDR_MOD_6); // 40 -> 64, no CLR_A

        // Rewind srcB to 0 for the next srcA tile (CLR_A clears srcA's dvalid); srcA keeps advancing.
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
    }

    // Clear srcB dvalid on the way out.
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB);

#undef BCAST_COLS_OP
}

// SDPA's SUB path keeps this op-named wrapper: its name is referenced directly by external callers
// (WH arch + the tt-llk eltwise_bcast_col tests), so it can't be folded away. The MUL
// path has no such external caller -- its API calls _llk_math_bcast_cols_reuse_custom_<ELWMUL> directly.
inline void _llk_math_sub_bcast_cols_reuse_custom_(const std::uint32_t ct_dim = 1)
{
    _llk_math_bcast_cols_reuse_custom_<EltwiseBinaryType::ELWSUB>(ct_dim);
}
