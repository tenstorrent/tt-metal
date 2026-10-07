// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include <algorithm>
#include <array>

#include <tt-metalium/runtime_args_data.hpp>
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"

#include "all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherProgramArtifacts {
    // Cache the binding objects, not their payload pointers: dispatch may relocate data().
    std::reference_wrapper<tt::tt_metal::RuntimeArgsData> reader_common_args;
    std::reference_wrapper<tt::tt_metal::RuntimeArgsData> writer_common_args;
    using RuntimeArgs = std::array<uint32_t, ttnn::ccl::AllGatherCommonArgs::count>;
    static RuntimeArgs collect_runtime_args(
        const std::optional<GlobalSemaphore>& barrier,
        const std::vector<GlobalSemaphore>& semaphores,
        const Tensor& input,
        const Tensor& output);

    void override_runtime_arguments(const RuntimeArgs& args) const {
        std::copy(args.begin(), args.end(), reader_common_args.get().data());
        std::copy(args.begin(), args.end(), writer_common_args.get().data());
    }
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
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn
