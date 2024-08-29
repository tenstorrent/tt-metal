// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/attn_matmul_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul {

    ttnn::Tensor AttnMatmulOperation::invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const CoreCoord& compute_with_storage_grid_size,
        std::optional<const DataType> dtype,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {

        auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
        auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
        return operation::run(AttnMatmulDeviceOperation{std::nullopt, std::nullopt, compute_with_storage_grid_size,
            memory_config.value_or(input_tensor_a.memory_config()), dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val},
            {input_tensor_a, input_tensor_b}, {}, {optional_output_tensor}, queue_id).at(0);
    }

    ttnn::Tensor AttnMatmulOperation::invoke(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const CoreCoord& compute_with_storage_grid_size,
        std::optional<const DataType> dtype,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {
        return invoke(
            ttnn::DefaultQueueId, input_tensor_a, input_tensor_b, compute_with_storage_grid_size, dtype, compute_kernel_config, memory_config, optional_output_tensor);
    }

    // TODO: Should we support option to read directly from cache (with optional transpose_hw)?
    ttnn::Tensor AttnMatmulFromCacheOperation::invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const uint32_t num_tokens,
        const bool transpose_hw,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<Tensor> optional_output_tensor) {

        TT_FATAL(num_tokens > 0, "Number of tokens must be at least 1!");
        TT_FATAL(num_tokens <= input_tensor_b.get_legacy_shape()[2], "Number of tokens must be smaller or equal to the max cache length (B.shape[2])!");
        const uint32_t num_tokens_rounded_up_to_32 = ((num_tokens - 1) / 32 + 1) * 32;
        auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
        auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
        return operation::run(AttnMatmulDeviceOperation{num_tokens_rounded_up_to_32,
            transpose_hw, compute_with_storage_grid_size, memory_config.value_or(input_tensor_a.memory_config()),
            dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val}, {input_tensor_a, input_tensor_b}, {}, {optional_output_tensor}, queue_id).at(0);
    }

    ttnn::Tensor AttnMatmulFromCacheOperation::invoke(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const uint32_t num_tokens,
        const bool transpose_hw,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<Tensor> optional_output_tensor) {
        return invoke(
            ttnn::DefaultQueueId, input_tensor_a, input_tensor_b, num_tokens, transpose_hw, compute_with_storage_grid_size, memory_config, dtype, compute_kernel_config, optional_output_tensor);
        }

};  // namespace ttnn::operations::experimental::matmul
