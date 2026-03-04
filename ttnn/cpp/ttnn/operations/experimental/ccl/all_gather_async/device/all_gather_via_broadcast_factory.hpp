// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherViaBroadcastFactoryProgramArtifacts {
    std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    tt::tt_metal::GlobalSemaphore semaphore;
    tt::tt_metal::GlobalSemaphore barrier_semaphore;
    uint32_t ring_index = 0;
};

struct AllGatherViaBroadcastFactory {
    // using shared_variables_t = AllBroadcastProgramFactory::shared_variables_t;
    using shared_variables_t = AllGatherViaBroadcastFactoryProgramArtifacts;
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
        const ttnn::MeshCoordinate& sender_device_coord,
        const Tensor& input,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& semaphore,
        const tt::tt_metal::GlobalSemaphore& barrier_semaphore);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {
using AllGatherViaBroadcastFactoryProgramArtifacts = experimental::prim::AllGatherViaBroadcastFactoryProgramArtifacts;

// Builder function that creates kernels and returns artifacts
AllGatherViaBroadcastFactoryProgramArtifacts build_all_gather_via_broadcast_program_artifacts(
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
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);


}  // namespace ttnn
