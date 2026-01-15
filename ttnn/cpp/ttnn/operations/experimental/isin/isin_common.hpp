// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"

#include <cstdint>

namespace ttnn::operations::experimental::isin::common {

constexpr int32_t FIRST_DIMENSION = 0;

constexpr auto OUTPUT_TENSOR_DATA_TYPE = tt::tt_metal::DataType::UINT32;
constexpr auto OUTPUT_TENSOR_LAYOUT = tt::tt_metal::Layout::ROW_MAJOR;
constexpr uint32_t OUTPUT_TENSOR_RANK = 1;

}  // namespace ttnn::operations::experimental::isin::common
