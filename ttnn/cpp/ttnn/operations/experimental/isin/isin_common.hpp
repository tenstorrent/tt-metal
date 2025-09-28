// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::isin::common {

using namespace tt::tt_metal;

constexpr int32_t FIRST_DIMENSION = 0;

constexpr auto OUTPUT_TENSOR_DATA_TYPE = DataType::UINT32;
constexpr auto OUTPUT_TENSOR_LAYOUT = Layout::ROW_MAJOR;
constexpr uint32_t OUTPUT_TENSOR_RANK = 1;

}  // namespace ttnn::operations::experimental::isin::common
