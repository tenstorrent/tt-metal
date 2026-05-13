// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

inline void _llk_math_eltwise_sfpu_done_with_addrmod_reset_()
{
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    // math::clear_addr_mod_base();
    TTI_SETC16(2, 0); // clear addr mod base (use addr mods 0..3)
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

template <DstSync Dst, bool Accum>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles<Dst, Accum, DstTileShape::Tile32x32>()), message);
}

inline void _llk_math_eltwise_sfpu_inc_vector_mode_dst_face_addr_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}

template <typename Callable, typename IncDstFaceAddr>
inline void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, IncDstFaceAddr&& inc_dst_face_addr, int vector_mode)
{
    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)();
            inc_dst_face_addr();
        }
        // Skip the next 2 faces
        inc_dst_face_addr();
        inc_dst_face_addr();
    }
    else if (mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)();
            inc_dst_face_addr();
            inc_dst_face_addr();
        }
    }
    else if (mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)();
            inc_dst_face_addr();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)();
    }
}
