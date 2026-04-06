// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/split_query_key_value_and_split_heads_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    uint32_t num_heads = 16,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);

}  // namespace ttnn::experimental
