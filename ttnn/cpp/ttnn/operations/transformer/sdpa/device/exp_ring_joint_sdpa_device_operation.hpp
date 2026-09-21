// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"

namespace ttnn::prim {

// The default hash is load-bearing: logical_n_tensor presence must reach it -- the kernels compile
// differently when set. Any future custom compute_program_hash must include that presence bit.
struct ExpRingJointSDPADeviceOperation {
    using operation_attributes_t = ExpRingJointSDPAParams;
    using tensor_args_t = ExpRingJointSDPAInputs;
    using spec_return_value_t = ExpRingJointSDPAResultSpec;
    using tensor_return_value_t = ExpRingJointSDPAResult;
    using program_factory_t = std::variant<ExpRingJointSDPAMeshWorkloadFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);
};

ExpRingJointSDPAResult exp_ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t num_workers_per_link = 1,
    uint32_t num_buffers_per_channel = 8,
    // When set, logical_n above is the worst-case placeholder and the live value is read on-device.
    const std::optional<ttnn::Tensor>& logical_n_tensor = std::nullopt);

}  // namespace ttnn::prim
