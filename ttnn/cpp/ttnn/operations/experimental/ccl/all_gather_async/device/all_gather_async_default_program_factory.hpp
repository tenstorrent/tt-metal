// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t num_directions_per_link = 0;
    uint32_t num_workers_per_direction = 0;
    uint32_t num_mux_cores_per_direction_per_link = 0;
    uint32_t num_cores_per_link = 0;
};

struct DefaultMeshWorkloadFactory {
    using shared_variables_t = AllGatherProgramArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {
using AllGatherProgramArtifacts = experimental::prim::AllGatherProgramArtifacts;

// Resolved per-op values for the single-batch-slice and partial (overallocated) gather features.
// Computed once from the input tensor + attributes and shared between program build (compile/runtime
// arg emission) and override_runtime_arguments (re-patch on cache hit) so the two never disagree.
struct AllGatherSliceParams {
    uint32_t batch_head_size = 0;   // batch*head loop count; reduced to one batch's heads when slicing
    uint32_t input_batch_base = 0;  // page offset into the input to the sliced batch slot (0 = full batch)
    uint32_t valid_pages = 0;       // per-batch-head valid page count; == full page count when not clamping
};

// Resolve the slice/partial-gather values. batch_slice_idx selects one index along dim 0 (requires the
// gather dim != 0); valid_gather_extent is the valid extent (elements, tile-aligned) along the gather
// dim, which must be the height dim (rank-2). Both default to std::nullopt => feature off (inert values).
AllGatherSliceParams compute_all_gather_slice_params(
    const Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> batch_slice_idx,
    std::optional<uint32_t> valid_gather_extent);

// Builder function that creates kernels and returns artifacts
AllGatherProgramArtifacts build_all_gather_async_minimal_default_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    int32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> batch_slice_idx = std::nullopt,
    std::optional<uint32_t> valid_gather_extent = std::nullopt);

// Runtime argument override function
void all_gather_async_minimal_default_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& output,
    uint32_t input_batch_base = 0,
    uint32_t valid_pages = std::numeric_limits<uint32_t>::max());

}  // namespace ttnn
