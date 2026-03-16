// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    const uint32_t num_heads,
    const std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors) {
    auto result = ttnn::prim::split_query_key_value_and_split_heads(
        input_tensor, compute_with_storage_grid_size, memory_config, num_heads, optional_output_tensors);
    return {result.at(0), result.at(1), result.at(2)};
}

}  // namespace ttnn::experimental
