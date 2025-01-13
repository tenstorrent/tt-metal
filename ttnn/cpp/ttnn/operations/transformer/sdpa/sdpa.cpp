// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa.hpp"

#include <utility>

#include "device/sdpa_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = is_causal,
                   .chunk_start_idx = std::nullopt,
                   .compute_kernel_config = kernel_config_val},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {attn_mask},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::move(attn_mask),
        is_causal,
        scale,
        memory_config,
        std::move(program_config),
        compute_kernel_config);
}

ttnn::Tensor ExecuteChunkedScaledDotProductAttention::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = true,  // Always causal for chunked version
                   .chunk_start_idx = chunk_start_idx,
                   .compute_kernel_config = kernel_config_val},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {std::nullopt, page_table_tensor},  // No attention mask - handled internally based on chunk_start_idx
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteChunkedScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        chunk_start_idx,
        scale,
        memory_config,
        std::move(program_config),
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
