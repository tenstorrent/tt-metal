// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::matmul {

// TODO: Group attention matmul will support sharding, mcasting, and should be faster; we should make attn_matmul (ie.
// KV heads = 1) a special case of group_attn_matmul and run the same op
struct GroupAttnMatmulOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> output_dtype = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::experimental::matmul

namespace experimental {

constexpr auto group_attn_matmul = ttnn::register_operation<
    "ttnn::experimental::group_attn_matmul",
    ttnn::operations::experimental::matmul::GroupAttnMatmulOperation>();

}  // namespace experimental

}  // namespace ttnn
