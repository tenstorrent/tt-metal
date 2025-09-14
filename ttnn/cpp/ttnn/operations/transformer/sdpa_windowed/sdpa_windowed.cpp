// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_windowed.hpp"

#include <utility>

#include "device/sdpa_windowed_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteWindowedScaledDotProductAttention::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& cu_window_seqlens,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return tt::tt_metal::operation::run(
               WindowedScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .compute_kernel_config = kernel_config_val},
               {input_tensor_q, input_tensor_k, input_tensor_v, cu_window_seqlens},
               {},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteWindowedScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& cu_window_seqlens,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cu_window_seqlens,
        scale,
        memory_config,
        std::move(program_config),
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
