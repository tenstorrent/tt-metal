// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_op.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_op.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async_deprecated/ring_attention_all_gather_async_op.hpp"
#include "ttnn/device.hpp"
#include <utility>

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
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
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
        {});

    return {results.at(0), results.at(1)};
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
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
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
    int32_t rank = input_tensor_k.logical_shape().rank();
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
        {});

    return {results.at(0), results.at(1), results.at(2)};
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
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
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
        std::move(program_config),
        kernel_config_val);
}

}  // namespace ttnn::operations::transformer
