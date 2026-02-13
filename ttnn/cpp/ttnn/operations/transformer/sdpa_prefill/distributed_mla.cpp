// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla.hpp"
#include "device/distributed_mla_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

ttnn::Tensor ExecuteDistributedMLA::invoke(
    const ttnn::Tensor& q_tensor,
    const ttnn::Tensor& k_tensor,
    const ttnn::Tensor& v_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<float> scale) {
    auto output_memory_config = memory_config.value_or(q_tensor.memory_config());

    return ttnn::prim::distributed_mla(q_tensor, k_tensor, v_tensor, cluster_axis, output_memory_config, scale);
}

}  // namespace ttnn::operations::transformer::sdpa_prefill

namespace ttnn::prim {

ttnn::Tensor distributed_mla(
    const ttnn::Tensor& q_tensor,
    const ttnn::Tensor& k_tensor,
    const ttnn::Tensor& v_tensor,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    std::optional<float> scale) {
    // Map simple API to full device arguments with defaults
    operations::transformer::sdpa_prefill::DistributedMlaSDPAParams operation_attributes{
        .device_order = 0,  // Will be calculated in program factory from mesh coordinate
        .cluster_axis = cluster_axis,
        .scale = scale,
        .output_mem_config = memory_config,
        .program_config = std::nullopt,  // Use defaults
        .compute_kernel_config = ttnn::DeviceComputeKernelConfig{},
        .chunk_start_idx = std::nullopt,     // No prefix caching
        .is_causal = true,                   // Enable causal masking for sequence distribution
        .use_mla = false,                    // Standard attention
        .head_dim_v = std::nullopt,          // Use defaults
        .sliding_window_size = std::nullopt  // Use defaults
    };

    operations::transformer::sdpa_prefill::DistributedMlaSDPAInputs tensor_args{
        .q = q_tensor,                          // Q tensor (sharded per device)
        .k = k_tensor,                          // K tensor (full sequence)
        .v = v_tensor,                          // V tensor (full sequence)
        .attn_mask = std::nullopt,              // Auto-generate causal mask
        .page_table = std::nullopt,             // No paging
        .attention_sink = std::nullopt,         // No attention sink
        .chunk_start_idx_tensor = std::nullopt  // No flexible chunking
    };

    // Execute the device operation
    using OperationType = operations::transformer::sdpa_prefill::DistributedMLADeviceOperation;
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
