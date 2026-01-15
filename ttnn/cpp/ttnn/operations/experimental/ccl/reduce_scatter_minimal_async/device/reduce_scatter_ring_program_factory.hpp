// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

// Use ReduceScatterProgramArtifacts as the shared variables type for consistency
using RingReduceScatterSharedVariables = ReduceScatterProgramArtifacts;

struct RingReduceScatterMeshWorkloadFactory {
    using shared_variables_t = RingReduceScatterSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

// Builder function for ring topology - creates program artifacts
ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
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
    CoreCoord core_grid_offset);

// Override runtime arguments helper for ring topology
void ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output);

}  // namespace ttnn::experimental::prim
