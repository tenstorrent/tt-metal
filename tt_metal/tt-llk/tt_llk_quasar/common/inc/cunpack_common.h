// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::unpack
{
// Number of rows for Unpack functions
constexpr static std::uint32_t UNPACR_STRIDE_MAX_ROWS = 8;
constexpr static std::uint32_t TRISC_ID               = 0;
} // namespace ckernel::unpack
