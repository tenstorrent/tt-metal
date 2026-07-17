// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dit_fused_distributed_rmsnorm_device_operation_types.hpp"
#include "dit_fused_distributed_rmsnorm_program_factory.hpp"

namespace ttnn::experimental::prim {

// Single-program fused distributed RMSNorm for Wan2.2 attention:
//   pre RMSNorm stats  +  ring AG of stats  +  post RMSNorm with optional head-split,
//   RoPE, and output dtype cast — all in one kernel program with the input
//   tensor kept L1-resident across the pre and post stages.
struct DitFusedDistributedRmsnormDeviceOperation {
    using operation_attributes_t = DitFusedDistributedRmsnormParams;
    using tensor_args_t = DitFusedDistributedRmsnormInputs;
    // [0] = user-visible output tensor.
    // [1] = persistent stats DRAM scratch (only when use_mux). Allocated by
    //       the framework via create_device_tensor, so the underlying
    //       MeshBuffer is allocated in lock-step across the mesh and every
    //       chip sees the same DRAM address — required for the fabric mcast
    //       to land at a consistent remote-chip page.
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DitFusedDistributedRmsnormMeshWorkloadFactory>;
    using shared_variables_t = DitFusedDistributedRmsnormMeshWorkloadFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Returns just the user-visible output (element 0 of the underlying
// tensor_return_value_t vector — the stats DRAM scratch is internal).
Tensor dit_fused_distributed_rmsnorm(
    const Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    float epsilon,
    uint32_t num_heads_per_device,
    bool per_head_norm,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
    ttnn::experimental::DitFusedNormType norm_type = ttnn::experimental::DitFusedNormType::RMS,
    const std::optional<const Tensor>& reciprocals = std::nullopt);

}  // namespace ttnn::prim
