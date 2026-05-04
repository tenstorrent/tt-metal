// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations::pool {

Tensor global_avg_pool2d(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace operations::pool

using ttnn::operations::pool::global_avg_pool2d;

}  // namespace ttnn
