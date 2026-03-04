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
struct AttnMatmulOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const CoreCoord& compute_with_storage_grid_size,
        std::optional<const DataType> output_dtype = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

struct AttnMatmulFromCacheOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        uint32_t num_tokens,
        bool transpose_hw,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::experimental::matmul

namespace experimental {

constexpr auto attn_matmul = ttnn::register_operation<
    "ttnn::experimental::attn_matmul",
    ttnn::operations::experimental::matmul::AttnMatmulOperation>();

constexpr auto attn_matmul_from_cache = ttnn::register_operation<
    "ttnn::experimental::attn_matmul_from_cache",
    ttnn::operations::experimental::matmul::AttnMatmulFromCacheOperation>();

}  // namespace experimental

}  // namespace ttnn
