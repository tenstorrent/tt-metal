// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_compat.h"

namespace ckernel::sfpu
{

sfpi_inline sfpi::vBool _sfpu_is_fp16_zero_(const sfpi::vFloat& value)
{
    return compat::fp_eq(value, 0.0f);
}

} // namespace ckernel::sfpu
