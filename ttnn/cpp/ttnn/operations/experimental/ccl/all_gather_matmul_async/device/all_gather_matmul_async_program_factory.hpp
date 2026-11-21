// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/distributed/types.hpp"

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

namespace detail {

struct AllGatherMatmulAsyncDescriptorAdapterOperation {
    using operation_attributes_t = AllGatherMatmulAsyncParams;
    using tensor_args_t = AllGatherMatmulAsyncInputs;
    using spec_return_value_t = AllGatherMatmulAsyncResultSpec;
    using tensor_return_value_t = AllGatherMatmulAsyncResult;
};

}  // namespace detail

// One ProgramDescriptor per mesh coordinate (ring index, neighbours and fabric connections are per device): the
// minimal default all-gather plus the 1D (mcast_in0) or 2D mcast matmul that consumes its output as it arrives,
// signalling through MatmulFusedOpSignaler / AllGatherFusedOpSignaler.
struct AllGatherMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherMatmulAsyncParams& operation_attributes,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Tensor addresses are buffer bindings and are patched by the framework on a cache hit. The all-gather's
// GlobalSemaphores are passed in per call (callers cycle them), are not part of the program hash, and so are
// re-applied here on every hit: a WorkloadDescriptor op has no per-Program override hook.
struct AllGatherMatmulAsyncMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::AllGatherMatmulAsyncDescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<AllGatherMatmulAsyncProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherMatmulAsyncParams& operation_attributes,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<AllGatherMatmulAsyncMeshWorkloadFactory>);

}  // namespace ttnn::experimental::prim
