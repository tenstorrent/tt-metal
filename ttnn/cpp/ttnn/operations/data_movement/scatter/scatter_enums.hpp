// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <type_traits>

#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::data_movement::scatter {

using namespace tt;

// supported reduction methods for scatter to be applied for source values coming from recurring indices - amin and amax
// currently not developed (torch doesn't support them)
enum class ScatterReductionType : uint8_t { INVALID, ADD, MULTIPLY, AMIN, AMAX };

}  // namespace ttnn::operations::data_movement::scatter
