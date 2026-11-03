// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

namespace detail {

struct SparseRingSDPADescriptorAdapterOperation {
    using operation_attributes_t = SparseRingSDPAParams;
    using tensor_args_t = SparseRingSDPAInputs;
    using spec_return_value_t = SparseRingSDPAResultSpec;
    using tensor_return_value_t = SparseRingSDPAResult;
};

}  // namespace detail

struct SparseRingSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SparseRingSDPAParams& args,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct SparseRingSDPAMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::SparseRingSDPADescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<SparseRingSDPAProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseRingSDPAParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SparseRingSDPAParams& args,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<SparseRingSDPAMeshWorkloadFactory>);

}  // namespace ttnn::prim
