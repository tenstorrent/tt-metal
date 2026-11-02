// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "reduce_scatter_minimal_async_descriptor_adapter.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

// One ProgramDescriptor per mesh coordinate (ring index, neighbours and fabric connections are per device).
struct RingReduceScatterProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// The caller-owned GlobalSemaphores are not part of the program hash (see ReduceScatterMinimalAsyncParams), so
// override_runtime_arguments re-applies them on every cache hit. It re-writes the whole common-args block by role,
// tensors included: the input may alias the persistent intermediate on the call that built the program, and the
// bindings resolved on that call cannot tell the two apart.
struct RingReduceScatterMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::ReduceScatterMinimalAsyncDescriptorAdapterOperation>::
        DescriptorMeshWorkloadAdapter<RingReduceScatterProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<RingReduceScatterMeshWorkloadFactory>);

ReduceScatterDescriptorArtifacts build_ring_reduce_scatter_minimal_async_program_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

// Re-writes the barrier / semaphore common-arg slots of a reduce-scatter reader and writer built by
// build_ring_reduce_scatter_minimal_async_program_descriptor or
// build_line_reduce_scatter_minimal_async_program_descriptor.
void apply_reduce_scatter_semaphore_args(
    tt::tt_metal::Program& program,
    const ReduceScatterDescriptorArtifacts& artifacts,
    ttnn::ccl::Topology topology,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore);

}  // namespace ttnn::experimental::prim
