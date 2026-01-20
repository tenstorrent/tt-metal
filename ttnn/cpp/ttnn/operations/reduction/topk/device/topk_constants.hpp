// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::prim::constants {

constexpr uint32_t multi_core_min_width = 8192;
constexpr uint32_t min_dim_per_core = 64;

}  // namespace ttnn::prim::constants
