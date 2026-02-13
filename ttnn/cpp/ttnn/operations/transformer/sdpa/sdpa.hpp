// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace transformer {

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt);

/// Chunked SDPA over paged K/V: one Q chunk per call, K/V in paged layout.
/// Two overloads: legacy (chunk_start_idx as int) or flexible (chunk_start_idx_tensor on device).

/// Legacy: chunk start index as scalar.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,  // Must be a multiple of program_config.q_chunk_size
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

/// Flexible: chunk start index in device tensor [1] (int32). Read at runtime; use for trace.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    SDPAProgramConfig program_config,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor ring_distributed_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id =
        std::nullopt,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<SDPAProgramConfig>& program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& page_table = std::nullopt,
    std::optional<int64_t> chunk_start_idx = std::nullopt);

}  // namespace transformer

}  // namespace ttnn
