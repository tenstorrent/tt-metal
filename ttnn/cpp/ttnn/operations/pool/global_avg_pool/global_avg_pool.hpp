// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operation.hpp"

namespace tt::tt_metal {

enum class PoolType { AVG };

Tensor global_avg_pool2d(
    const Tensor& input,
    const MemoryConfig& memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace tt::tt_metal

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations::pool {

Tensor global_avg_pool2d(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace operations::pool

using ttnn::operations::pool::global_avg_pool2d;

}  // namespace ttnn
