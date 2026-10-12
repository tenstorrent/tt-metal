// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_kv_gather.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// One Program per mesh coordinate (the device's ring position, neighbours and per-chain shard counts depend on
// it): the VSA streaming descriptor is materialized into a Program, then the op's own K/V gather
// (build_vsa_kv_gather) adds the sender cores that forward this device's K and V shards around the ring,
// token-major, and signal the VSA leaders per landed shard.
struct VsaRingSdpaMeshWorkloadFactory {
    using shared_variables_t = VsaKvGatherArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const VsaRingSdpaParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);

    // Cache hit: the gather's override re-applies its buffer and GlobalSemaphore addresses (excluded from the
    // program hash); the VSA patch re-applies the VSA kernels' addresses and the gathered/semaphore addresses.
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<VsaRingSdpaMeshWorkloadFactory>);

// The RingAttention gather (the mergeable default): the stock ring_attention_all_gather_async helper is appended to
// the VSA program DESCRIPTOR by the builder, and the descriptor adapter re-binds every kernel's buffer addresses on
// cache hits (the helper keeps no kernel handles); the override then re-applies the VSA kernels' raw address args,
// the ring common args and the all-gather's GlobalSemaphore addresses.
namespace detail {
struct VsaRingSdpaDescriptorAdapterOperation {
    using operation_attributes_t = VsaRingSdpaParams;
    using tensor_args_t = VsaRingSdpaInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
};
}  // namespace detail

struct VsaRingSdpaRaProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct VsaRingSdpaRaMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::VsaRingSdpaDescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<VsaRingSdpaRaProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const VsaRingSdpaParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<VsaRingSdpaRaMeshWorkloadFactory>);

}  // namespace ttnn::prim
