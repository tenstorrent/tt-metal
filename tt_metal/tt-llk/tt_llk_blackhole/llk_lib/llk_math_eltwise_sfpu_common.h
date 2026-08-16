// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_sfpu_done_()
{
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    // Typed architectural Dst face advance: two CR-mode Dst += 8 counter steps,
    // expressed with the typed TTINCRWC builtin so the compiler's region and
    // ownership proofs see a typed Dst/RWC effect instead of an opaque
    // `.ttinsn` word (migration idiom of df504b3b2).  This branch's compiler
    // head exposes no rvtt_ttdstface / rvtt_ttsetrwc builtin, so the raw
    // SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D) word is replaced by the
    // architecturally equivalent CR-mode INCRWC (dst_rwc_cr += 8; dst_rwc =
    // dst_rwc_cr under both encodings), an established production idiom --
    // cf. TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0) in
    // llk_math_eltwise_binary.h.
    //
    // The builtin range-checks its increment as a signed 4-bit field, so the
    // unsigned field value 8 must be spelled -8; the assembled word is
    // byte-identical to raw TT_OP_INCRWC(p_setrwc::CR_D, 8, 0, 0)
    // (`ttincrwc 4,8,0,0`, 0xe0480000).
    __builtin_rvtt_ttincrwc(p_setrwc::CR_D, -8, 0, 0);
    __builtin_rvtt_ttincrwc(p_setrwc::CR_D, -8, 0, 0);
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

template <DstSync Dst, bool Accum>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles<Dst, Accum, DstTileShape::Tile32x32>()), message);
}

template <typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    if (vector_mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else if (vector_mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        // Skip the next 2 faces
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    else if (vector_mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}
