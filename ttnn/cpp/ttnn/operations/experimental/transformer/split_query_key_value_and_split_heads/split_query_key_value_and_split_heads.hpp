// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/split_query_key_value_and_split_heads_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct SplitFusedQKVAndSplitHeadsOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const uint32_t num_heads = 16,
        const std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors = std::nullopt) {
        auto result = ttnn::prim::split_query_key_value_and_split_heads(
            input_tensor, compute_with_storage_grid_size, memory_config, num_heads, optional_output_tensors);
        return {result.at(0), result.at(1), result.at(2)};
    }
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto split_query_key_value_and_split_heads = ttnn::register_operation<
    "ttnn::experimental::split_query_key_value_and_split_heads",
    ttnn::operations::experimental::transformer::SplitFusedQKVAndSplitHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
