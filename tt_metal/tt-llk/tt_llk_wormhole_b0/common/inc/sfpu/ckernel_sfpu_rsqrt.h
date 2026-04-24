// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_rsqrt_compat.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat = false>
inline void _calculate_rsqrt_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, int iterations)
{
    if constexpr (legacy_compat)
    {
        return _calculate_rsqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(dst_index_in, dst_index_out, iterations);
    }
    else
    {
        return _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>(dst_index_in, dst_index_out, iterations);
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_rsqrt_()
{
    if constexpr (!legacy_compat)
    {
        _init_sqrt_<APPROXIMATION_MODE>();
    }
}

} // namespace sfpu
} // namespace ckernel
