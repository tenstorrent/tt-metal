// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"

namespace ttnn::experimental::prim {

namespace detail {

struct MatmulReduceScatterAsyncDescriptorAdapterOperation {
    using operation_attributes_t = MatmulReduceScatterAsyncParams;
    using tensor_args_t = MatmulReduceScatterAsyncInputs;
    using spec_return_value_t = MatmulReduceScatterAsyncResultSpec;
    using tensor_return_value_t = MatmulReduceScatterAsyncResult;
};

}  // namespace detail

// One ProgramDescriptor per mesh coordinate (ring index, neighbours and fabric connections are per device). Each
// program is the ring reduce-scatter followed by the 2D mcast matmul that feeds it, signalling through
// MatmulFusedOpSignaler / ReduceScatterFusedOpSignaler.
struct MatmulReduceScatterAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MatmulReduceScatterAsyncParams& args,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Tensor addresses are buffer bindings and are patched by the framework on a cache hit. The reduce-scatter's
// GlobalSemaphores are passed in per call (callers cycle them), are not part of the program hash, and so are
// re-applied here on every hit: a WorkloadDescriptor op has no per-Program override hook.
struct MatmulReduceScatterAsyncMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::MatmulReduceScatterAsyncDescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<MatmulReduceScatterAsyncProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulReduceScatterAsyncParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MatmulReduceScatterAsyncParams& args,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<MatmulReduceScatterAsyncMeshWorkloadFactory>);

}  // namespace ttnn::experimental::prim
