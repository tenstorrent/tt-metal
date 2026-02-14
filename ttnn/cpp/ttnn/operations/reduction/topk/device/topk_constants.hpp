// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::prim::constants {

constexpr uint32_t multi_core_min_width = 8192;  // Minimum width to consider multi-core execution
constexpr uint32_t min_dim_per_core = 64;        // Minimum dimension size per core required

}  // namespace ttnn::prim::constants
