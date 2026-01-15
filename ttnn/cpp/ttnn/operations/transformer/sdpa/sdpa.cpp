// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& attention_sink) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask,
        std::nullopt,  // page_table
        attention_sink,
        is_causal,
        scale,
        sliding_window_size,
        std::nullopt,  // chunk_start_idx
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
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
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,        // attn_mask
        page_table_tensor,   // page_table
        std::nullopt,        // attention_sink
        /*is_causal=*/true,  // Always causal for chunked version
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ExecuteJointAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    SDPAProgramConfig program_config,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_tensors = ttnn::prim::joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        joint_strategy,
        program_config,
        scale,
        compute_kernel_config);
    return {output_tensors.output, output_tensors.joint_output};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteRingJointAttention::invoke(
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
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        joint_strategy,
        logical_n,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        ccl_core_grid_offset,
        subdevice_id,
        scale,
        compute_kernel_config);
    return {output_tensors.output, output_tensors.joint_output, output_tensors.lse_output};
}

ttnn::Tensor ExecuteFlashMLAPrefill::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,  // V is implied by K in MLA mode
        attn_mask,
        std::nullopt,  // page_table
        std::nullopt,  // attention_sink
        is_causal,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        std::nullopt,  // chunk_start_idx
        true,          // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor ExecuteChunkedFlashMLAPrefill::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,       // V is implied by K in MLA mode
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        std::nullopt,       // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        true,  // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor ExecuteRingDistributedScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t>
        ring_id,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::ring_distributed_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        ring_size,
        ring_id,  // Pass through the ring_id parameter (can be used or ignored)
        scale,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        page_table,
        chunk_start_idx);
}

}  // namespace ttnn::operations::transformer
