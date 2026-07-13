// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dit_fused_distributed_groupnorm_device_operation_types.hpp"
#include "dit_fused_distributed_groupnorm_program_factory.hpp"

namespace ttnn::experimental::prim {

// Single-program fused distributed GroupNorm for spatially sharded activations:
//   PRE per-group (sum, sumsq, count)  →  fabric AG of stats sticks on
//   cluster_axis  →  merge  →  POST (x-μ)*rsqrt(σ²+eps)*γ+β.
//
// ring_size==1: local PRE+POST only (no fabric). ring_size>1: 1 worker + 1
// forwarder, max_rounds=1, stick_bytes = num_groups * 16.
struct DitFusedDistributedGroupnormDeviceOperation {
    using operation_attributes_t = DitFusedDistributedGroupnormParams;
    using tensor_args_t = DitFusedDistributedGroupnormInputs;
    // [0] = user-visible output.
    // [1] = persistent stats DRAM scratch (only when ring_size > 1).
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DitFusedDistributedGroupnormMeshWorkloadFactory>;
    using shared_variables_t = DitFusedDistributedGroupnormMeshWorkloadFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_fused_distributed_groupnorm(
    const Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool use_welford,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id);

}  // namespace ttnn::prim
