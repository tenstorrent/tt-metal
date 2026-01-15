// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/attn_matmul_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "attn_matmul.hpp"
#include "ttnn/device.hpp"
#include <utility>

namespace ttnn::operations::experimental::matmul {

ttnn::Tensor AttnMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    std::optional<const DataType> dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return ttnn::prim::attn_matmul(
        input_tensor_a,
        input_tensor_b,
        compute_with_storage_grid_size,
        dtype,
        compute_kernel_config,
        memory_config,
        std::nullopt,  // num_tokens
        std::nullopt,  // transpose_hw
        std::move(optional_output_tensor));
}

// TODO: Should we support option to read directly from cache (with optional transpose_hw)?
ttnn::Tensor AttnMatmulFromCacheOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t num_tokens,
    const bool transpose_hw,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(num_tokens > 0, "Number of tokens must be at least 1!");
    TT_FATAL(
        num_tokens <= input_tensor_b.padded_shape()[2],
        "Number of tokens must be smaller or equal to the max cache length (B.shape[2])!");
    const uint32_t num_tokens_rounded_up_to_32 = ((num_tokens - 1) / 32 + 1) * 32;
    return ttnn::prim::attn_matmul(
        input_tensor_a,
        input_tensor_b,
        compute_with_storage_grid_size,
        dtype,
        compute_kernel_config,
        memory_config,
        num_tokens_rounded_up_to_32,
        transpose_hw,
        std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::experimental::matmul
