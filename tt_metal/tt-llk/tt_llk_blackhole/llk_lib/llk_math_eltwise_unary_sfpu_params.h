// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef TT_SFPU_PARAMS_INTERNAL_USE
#error "Do not include llk_math_eltwise_unary_sfpu_params.h directly. Include llk_math_eltwise_unary_sfpu_macros.h and use SFPU_UNARY_CALL* macros instead."
#endif

#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

template <typename Callable, typename... Args>
[[deprecated("Use SFPU_UNARY_CALL, SFPU_CALL_MODE, SFPU_UNARY_CALL, or SFPU_CALL_CAST from llk_math_eltwise_unary_sfpu_macros.h instead.")]]
inline void _llk_math_eltwise_unary_sfpu_params_(Callable&& sfpu_func, std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(dst_index);

    _llk_math_eltwise_sfpu_apply_vector_mode_(std::forward<Callable>(sfpu_func), vector_mode, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}
