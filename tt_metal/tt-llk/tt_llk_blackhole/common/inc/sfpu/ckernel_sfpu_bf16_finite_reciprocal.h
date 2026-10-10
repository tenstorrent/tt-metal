// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace sfpi
{
template <typename Reciprocal>
inline vFloat correction_finite_reciprocal(vFloat x, [[maybe_unused]] Reciprocal reciprocal)
{
    vFloat y        = sfpi::approx_recip(x);
    vFloat residual = x * y - sfpi::vConstFloatPrgm0;
    return y * -residual - 0.0f;
}
} // namespace sfpi
