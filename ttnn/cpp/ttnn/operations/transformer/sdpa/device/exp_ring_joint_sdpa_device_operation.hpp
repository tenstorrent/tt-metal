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

struct ExpRingJointSDPADeviceOperation {
    using operation_attributes_t = ExpRingJointSDPAParams;
    using tensor_args_t = ExpRingJointSDPAInputs;
    using spec_return_value_t = ExpRingJointSDPAResultSpec;
    using tensor_return_value_t = ExpRingJointSDPAResult;
    using program_factory_t = std::variant<ExpRingJointSDPAProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);

    // The per-link GlobalSemaphore addresses are excluded from the program-cache hash
    // (ExpRingJointSDPAParams::attribute_values omits `semaphore`), so they are DYNAMIC: the factory
    // bakes them for the cache-miss build and this method re-applies them to the cached program on
    // every dispatch, so a cache hit with a different semaphore set cannot reuse a frozen address.
    // Slot layout mirrors the factory via the shared exp_ring_joint_sdpa_dynamic constants.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
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
    uint32_t num_buffers_per_channel = 8);

}  // namespace ttnn::prim
