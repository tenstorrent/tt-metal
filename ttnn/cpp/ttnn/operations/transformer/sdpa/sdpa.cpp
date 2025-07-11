// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa.hpp"

#include <utility>

#include "device/sdpa_op.hpp"
#include "device/joint_sdpa_op.hpp"
#include "device/ring_joint_sdpa_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_op.hpp"

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttention::invoke(
    QueueId queue_id,
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

    return tt::tt_metal::operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = is_causal,
                   .chunk_start_idx = std::nullopt,
                   .compute_kernel_config = kernel_config_val,
                   .use_mla = false},
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
    QueueId queue_id,
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

    return tt::tt_metal::operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = true,  // Always causal for chunked version
                   .chunk_start_idx = chunk_start_idx,
                   .compute_kernel_config = kernel_config_val,
                   .use_mla = false},
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

std::tuple<ttnn::Tensor, ttnn::Tensor> ExecuteJointAttention::invoke(
    QueueId queue_id,
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
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto results = tt::tt_metal::operation::run(
        JointScaledDotProductAttention{
            .joint_strategy = joint_strategy,
            .scale = scale,
            .output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            .program_config = std::move(program_config),
            .compute_kernel_config = kernel_config_val},
        {input_tensor_q, input_tensor_k, input_tensor_v, joint_tensor_q, joint_tensor_k, joint_tensor_v},
        {},
        {},
        queue_id);

    return {results.at(0), results.at(1)};
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
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        joint_strategy,
        std::move(program_config),
        scale,
        compute_kernel_config);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteRingJointAttention::invoke(
    QueueId queue_id,
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
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    /**
     * Create RingAttentionAllGatherAsync struct.
     * It will be a member of the RingJointScaledDotProductAttention struct.
     */
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API withou 2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensor_k.get_logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    auto all_gather_struct = ttnn::RingAttentionAllGatherAsync{
        {},
        gather_dim,
        num_links,
        num_devices,
        input_tensor_k.memory_config(),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis};

    const std::vector<ttnn::Tensor> input_tensors = {
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
    };

    auto results = tt::tt_metal::operation::run(
        RingJointScaledDotProductAttention{
            joint_strategy,
            scale,
            logical_n,
            num_devices, /* ring_size */
            tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            std::move(program_config),
            kernel_config_val,
            all_gather_struct,
            ccl_core_grid_offset},
        input_tensors,
        {},
        {},
        queue_id);

    return {results.at(0), results.at(1), results.at(2)};
}

ttnn::Tensor ExecuteFlashMLAPrefill::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
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

    return tt::tt_metal::operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = is_causal,
                   .chunk_start_idx = std::nullopt,
                   .compute_kernel_config = kernel_config_val,
                   .use_mla = true,
                   .head_dim_v = head_dim_v},
               {input_tensor_q, input_tensor_k},
               {attn_mask},
               {},
               queue_id)
        .at(0);
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
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        head_dim_v,
        std::move(attn_mask),
        is_causal,
        scale,
        memory_config,
        std::move(program_config),
        compute_kernel_config);
}

ttnn::Tensor ExecuteChunkedFlashMLAPrefill::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
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

    return tt::tt_metal::operation::run(
               ScaledDotProductAttention{
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = std::move(program_config),
                   .is_causal = true,  // Always causal for chunked version
                   .chunk_start_idx = chunk_start_idx,
                   .compute_kernel_config = kernel_config_val,
                   .use_mla = true,
                   .head_dim_v = head_dim_v},
               {input_tensor_q, input_tensor_k},
               {std::nullopt, page_table_tensor},  // No attention mask - handled internally based on chunk_start_idx
               {},
               queue_id)
        .at(0);
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
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        head_dim_v,
        page_table_tensor,
        chunk_start_idx,
        scale,
        memory_config,
        std::move(program_config),
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
