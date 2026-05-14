// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_sfpu_common.h"

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(0);

    auto invoke_sfpu = [&]() { std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...); };
    _llk_math_eltwise_sfpu_apply_vector_mode_(invoke_sfpu, vector_mode);

    _llk_math_eltwise_sfpu_done_();
}
