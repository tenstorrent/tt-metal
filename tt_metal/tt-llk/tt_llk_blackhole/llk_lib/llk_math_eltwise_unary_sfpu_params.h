// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_(dst_index);

    auto invoke_sfpu = [&]() { std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...); };
    _llk_math_eltwise_sfpu_apply_vector_mode_(invoke_sfpu, _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_, vector_mode);

    _llk_math_eltwise_unary_sfpu_done_();
}
