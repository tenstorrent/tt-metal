// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// True for +0.0 and -0.0. A bare `v == 0.0F` lowers to SFPSETCC LREG_EQ0, an all-32-bits-zero test,
// so -0.0 (0x80000000) would read as nonzero; clearing the sign first costs one SFPABS.
sfpi_inline sfpi::vBool _sfpu_is_fp16_zero_(const sfpi::vFloat& v)
{
    return sfpi::abs(v) == 0.0F;
}

} // namespace sfpu
} // namespace ckernel
