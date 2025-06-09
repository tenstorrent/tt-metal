// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <type_traits>

#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::experimental::scatter {

using namespace tt;

enum class ScatterReductionType : uint8_t { ADD, MULTIPLY, AMIN, AMAX };

enum class ProcessingMode : uint8_t {
    FULL_ROW_PROCESSING,
    SPLIT_ROW_PROCESSING
};

}  // namespace ttnn::operations::experimental::scatter
