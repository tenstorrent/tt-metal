// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_sfpu.h"
#include "llk_defs.h"

using namespace ckernel;
using namespace ckernel::math;

inline void _eltwise_sfpu_configure_addrmod_()
{
    _sfpu_configure_addrmod_();
}

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t tile_index)
{
    _llk_math_sfpu_start_(tile_index);
}

inline void _llk_math_eltwise_sfpu_done_()
{
    _llk_math_sfpu_done_();
}

template <bool SRCS_RD_DONE, bool SRCS_WR_DONE>
inline void _llk_math_eltwise_sfpu_srcs_clear_vlds_()
{
    _llk_math_sfpu_srcs_clear_vlds_<SRCS_RD_DONE, SRCS_WR_DONE>();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    _llk_math_sfpu_inc_dst_face_addr_();
}

inline void _llk_math_eltwise_sfpu_init_()
{
    _llk_math_sfpu_init_();
}

template <class F, class... ARGS>
inline void _llk_math_eltwise_sfpu_params_(F&& sfpu_func, std::uint32_t dst_tile_index, ARGS&&... args)
{
    _llk_math_sfpu_params_(std::forward<F>(sfpu_func), dst_tile_index, std::forward<ARGS>(args)...);
}
