// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h, ported except SFPU. Same Tensix instruction
// sequence as production. Also holds the ADDR_MOD helper shared by the SFPU wrapper headers.

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

namespace hal_port
{

// addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = dest_incr}}.set(slot): three SETC16 words
// (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <cfg::Sec S, std::uint32_t dest_incr>
inline void set_sfpu_addr_mod()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// math::inc_dst_addr<8>(): TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D).
inline void inc_dst_addr_8()
{
    // HAL-GAP: rwc::set plans a lone Dest AdvanceAndReset as INCRWC; no way to request the SETRWC(CR_D, 8, SET_D) encoding
    ckernel::math::inc_dst_addr<8>();
}

} // namespace hal_port

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    hal::sync::wait::stall<hal::sync::StallTarget::Sfpu, hal::sync::StallCondition::MathIdle>();
}

inline void _llk_math_eltwise_sfpu_done_()
{
    hal_port::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    hal_port::inc_dst_addr_8();
    hal_port::inc_dst_addr_8();
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

template <DstSync Dst>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
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
