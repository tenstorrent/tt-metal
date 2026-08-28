// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct RingJointSDPADeviceOperation {
    using operation_attributes_t = RingJointSDPAParams;
    using tensor_args_t = RingJointSDPAInputs;
    using spec_return_value_t = RingJointSDPAResultSpec;
    using tensor_return_value_t = RingJointSDPAResult;
    using program_factory_t = std::variant<RingJointSDPAMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);
};

RingJointSDPAResult ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_v,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    std::size_t logical_l,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    bool is_causal = false,
    bool is_balanced = false,
    bool is_cross = false,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    std::optional<uint32_t> kv_cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_actual_isl = std::nullopt,
    std::optional<uint32_t> latent_v_head_dim = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    const std::optional<ttnn::Tensor>& slot_id = std::nullopt,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor = std::nullopt,
    uint32_t kv_cache_num_layers = 1,
    uint32_t kv_cache_layer_idx = 0,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    std::optional<uint32_t> tokens_per_frame = std::nullopt,
    std::optional<uint32_t> num_frames_padded = std::nullopt,
    std::vector<uint32_t> sparse_frame_mask = {},
    const std::optional<ttnn::Tensor>& reference_kv = std::nullopt,
    std::optional<uint32_t> reference_frame_idx = std::nullopt);

}  // namespace ttnn::prim
