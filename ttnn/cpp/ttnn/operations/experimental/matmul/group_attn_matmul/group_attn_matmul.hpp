// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// TODO: Group attention matmul will support sharding, mcasting, and should be faster; we should make attn_matmul (ie.
// KV heads = 1) a special case of group_attn_matmul and run the same op
Tensor group_attn_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn::experimental
