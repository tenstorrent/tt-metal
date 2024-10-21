// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa.hpp"

#include "device/sdpa_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    std::optional<ttnn::Tensor> attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                     : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .is_causal = is_causal,
                   .compute_kernel_config = kernel_config_val},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {attn_mask},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    std::optional<ttnn::Tensor> attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask,
        is_causal,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
