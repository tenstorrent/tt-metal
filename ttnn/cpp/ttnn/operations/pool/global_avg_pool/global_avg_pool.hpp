// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operation.hpp"

namespace tt {
namespace tt_metal {

enum class PoolType { AVG };

Tensor global_avg_pool2d(
    const Tensor& input,
    const MemoryConfig& memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace tt_metal
}  // namespace tt

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations {
namespace pool {

struct GlobalAveragePool2D {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DataType>& output_dtype = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input.memory_config());
        auto result = tt::tt_metal::global_avg_pool2d(input, memory_config, output_dtype);
        return result;
    }
};
}  // namespace pool
}  // namespace operations

constexpr auto global_avg_pool2d =
    ttnn::register_operation<"ttnn::global_avg_pool2d", ttnn::operations::pool::GlobalAveragePool2D>();

}  // namespace ttnn
