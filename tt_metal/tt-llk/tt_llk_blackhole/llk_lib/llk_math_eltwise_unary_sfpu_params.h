// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

// Dispatch the SFPU callable with or without leading dst indices, depending on
// which signature it accepts. Some calculate functions take (dst_in, dst_out,
// args...) — these are the split-aware ones added on top of the legacy
// (args...) ones (e.g. TopK helpers). This helper selects the right shape at
// compile time so the single-idx `_params_` template can drive both kinds of
// callable while keeping in == out for the legacy single-dst contract.
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_dispatch_(Callable&& sfpu_func, std::uint32_t dst_index, Args&&... args)
{
    if constexpr (std::is_invocable_v<Callable&, std::uint32_t, std::uint32_t, Args&...>)
    {
        std::forward<Callable>(sfpu_func)(dst_index, dst_index, std::forward<Args>(args)...);
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}

// Single-index variant. The dispatch helper above lets a callable that
// accepts (dst_in, dst_out, Args&...) work here by repeating dst_index for
// both positions (in == out), so legacy single-dst paths and macro callers
// can drive split-aware SFPU calculate functions without changing shape.
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(dst_index);

    auto dispatch_wrapper = [&](auto&&... a) {
        _llk_math_eltwise_unary_sfpu_dispatch_(
            std::forward<Callable>(sfpu_func), dst_index, std::forward<decltype(a)>(a)...);
    };
    _llk_math_eltwise_sfpu_apply_vector_mode_(dispatch_wrapper, vector_mode, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}

// Split-dest variant: source tile index (dst_index_in) is used to position
// the dest face pointer at the start; the callable then receives both indices
// so an SFPU op can read from dst_index_in and write to dst_index_out.
// Distinct name (not an overload of _params_) so callers pick the shape
// explicitly and overload resolution can't conflate the two.
// The dst-bound assert lives in ckernel::_sfpu_check_and_call_ (see the
// per-arch llk_math_eltwise_unary_sfpu_macros.h), so it is intentionally not
// duplicated here.
template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_split_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(dst_index_in);

    auto split_wrapper = [&](auto&&... a) {
        std::forward<Callable>(sfpu_func)(dst_index_in, dst_index_out, std::forward<decltype(a)>(a)...);
    };
    _llk_math_eltwise_sfpu_apply_vector_mode_(split_wrapper, vector_mode, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}
