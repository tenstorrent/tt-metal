// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "wan_fused_distributed_rmsnorm_device_operation_types.hpp"
#include "wan_fused_distributed_rmsnorm_program_factory.hpp"

namespace ttnn::experimental::prim {

// Single-program fused distributed RMSNorm for Wan2.2 attention:
//   pre RMSNorm stats  +  ring AG of stats  +  post RMSNorm with optional head-split,
//   RoPE, and output dtype cast — all in one kernel program with the input
//   tensor kept L1-resident across the pre and post stages.
struct WanFusedDistributedRmsnormDeviceOperation {
    using operation_attributes_t = WanFusedDistributedRmsnormParams;
    using tensor_args_t = WanFusedDistributedRmsnormInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<WanFusedDistributedRmsnormMeshWorkloadFactory>;
    using shared_variables_t = WanFusedDistributedRmsnormMeshWorkloadFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::WanFusedDistributedRmsnormDeviceOperation::tensor_return_value_t
wan_fused_distributed_rmsnorm(
    const Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    float epsilon,
    uint32_t num_heads_per_device,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
