// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::pool {
struct AdaptiveAvgPool2DOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config);
};

struct AdaptiveAvgPool1DOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config);
};

}  // namespace operations::pool

constexpr auto adaptive_avg_pool2d =
    ttnn::register_operation<"ttnn::adaptive_avg_pool2d", ttnn::operations::pool::AdaptiveAvgPool2DOperation>();

constexpr auto adaptive_avg_pool1d =
    ttnn::register_operation<"ttnn::adaptive_avg_pool1d", ttnn::operations::pool::AdaptiveAvgPool1DOperation>();

}  // namespace ttnn
