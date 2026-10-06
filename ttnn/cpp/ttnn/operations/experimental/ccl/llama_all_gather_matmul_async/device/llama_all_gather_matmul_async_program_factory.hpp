// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

namespace detail {

struct LlamaAllGatherMatmulAsyncDescriptorAdapterOperation {
    using operation_attributes_t = LlamaAllGatherMatmulAsyncParams;
    using tensor_args_t = LlamaAllGatherMatmulAsyncInputs;
    using spec_return_value_t = LlamaAllGatherMatmulAsyncResultSpec;
    using tensor_return_value_t = LlamaAllGatherMatmulAsyncResult;
};

}  // namespace detail

// One ProgramDescriptor per mesh coordinate: the llama all-gather (reader, writer, receiver) and the gather_in0 ring
// matmul it feeds through the LLAMA_ALL_GATHER MatmulFusedOpSignaler.
struct LlamaAllGatherMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const LlamaAllGatherMatmulAsyncParams& args,
        const LlamaAllGatherMatmulAsyncInputs& tensor_args,
        LlamaAllGatherMatmulAsyncResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Tensor addresses are buffer bindings patched by the framework on a cache hit; the caller-owned GlobalSemaphore is
// not in the program hash and is re-applied here (a WorkloadDescriptor op has no per-Program override hook).
struct LlamaAllGatherMatmulAsyncMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::LlamaAllGatherMatmulAsyncDescriptorAdapterOperation>::
        DescriptorMeshWorkloadAdapter<LlamaAllGatherMatmulAsyncProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const LlamaAllGatherMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const LlamaAllGatherMatmulAsyncInputs& tensor_args,
        LlamaAllGatherMatmulAsyncResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const LlamaAllGatherMatmulAsyncParams& args,
        const LlamaAllGatherMatmulAsyncInputs& tensor_args,
        LlamaAllGatherMatmulAsyncResult& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<LlamaAllGatherMatmulAsyncMeshWorkloadFactory>);

}  // namespace ttnn::experimental::prim
