// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

namespace detail {
struct VsaRingSdpaDescriptorAdapterOperation {
    using operation_attributes_t = VsaRingSdpaParams;
    using tensor_args_t = VsaRingSdpaInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
};
}  // namespace detail

// One ProgramDescriptor per mesh coordinate: the device's ring position, its neighbors and the number of
// shards each chain delivers all depend on the coordinate.
struct VsaRingSdpaProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct VsaRingSdpaMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::VsaRingSdpaDescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<VsaRingSdpaProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const VsaRingSdpaParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);

    // Cache hit: buffers are re-bound by the adapter; this re-applies the VSA kernels' raw address args and
    // the all-gather's GlobalSemaphore addresses (excluded from the program hash, so a hit with the other
    // ping-pong semaphore set must not keep the address frozen at the first miss).
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<VsaRingSdpaMeshWorkloadFactory>);

}  // namespace ttnn::prim
