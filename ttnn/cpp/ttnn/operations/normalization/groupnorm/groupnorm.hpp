// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

struct ExecuteGroupNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int num_groups,
        const float epsilon,
        const std::optional<ttnn::Tensor>& input_mask = std::nullopt,
        const std::optional<ttnn::Tensor>& weight = std::nullopt,
        const std::optional<ttnn::Tensor>& bias = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<CoreGrid> core_grid = std::nullopt,
        std::optional<bool> inplace = std::nullopt,
        std::optional<ttnn::Layout> output_layout = std::nullopt);
};

}  // namespace normalization
}  // namespace operations

constexpr auto group_norm = ttnn::
    register_operation_with_auto_launch_op<"ttnn::group_norm", ttnn::operations::normalization::ExecuteGroupNorm>();

}  // namespace ttnn
