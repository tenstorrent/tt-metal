// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::pool {
ttnn::Tensor compute_adaptive_avg_pool(
    const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config);

inline int64_t start_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
    return (out_idx * in_size) / out_size;
}

inline int64_t end_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
    return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}

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
