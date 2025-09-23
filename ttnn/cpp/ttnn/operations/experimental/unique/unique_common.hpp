// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "unique_cbs.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::unique::common {

using namespace tt;
using namespace tt_metal;

constexpr int32_t FIRST_DIMENSION = 0;
constexpr int32_t ONE_DIMENSION = 1;
constexpr uint32_t OUTPUT_TENSOR_RANK = 1;
constexpr uint32_t OUTPUT_TENSOR_SIZE = 64;
constexpr auto OUTPUT_TENSOR_LAYOUT = Layout::ROW_MAJOR;
constexpr auto OUTPUT_SIZE_TENSOR_DATA_TYPE = DataType::UINT32;
constexpr auto FIRST_OCCURRENCES_TENSOR_DATA_TYPE = DataType::UINT32;
constexpr auto OUTPUT_SIZE_TENSOR_SIZE = 64;

}  // namespace ttnn::operations::experimental::unique::common
