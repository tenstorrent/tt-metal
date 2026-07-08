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

sfpi_inline sfpi::vBool _sfpu_is_fp16_zero_(const sfpi::vFloat& v)
{
    return v == 0.0F;
}

} // namespace sfpu
} // namespace ckernel
