// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include <functional>

#include <algorithm>
#include <array>

#include <optional>

#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"

#include "all_gather_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Returned by build_all_gather_async_minimal_default_program_descriptor: where the all-gather's reader and writer
// landed in the caller's ProgramDescriptor (kernel indices double as kernel handles in a Program built from it).
struct AllGatherDescriptorArtifacts {
    size_t reader_kernel_index = 0;
    size_t writer_kernel_index = 0;
};

struct DefaultMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {
using AllGatherDescriptorArtifacts = experimental::prim::AllGatherDescriptorArtifacts;

// Appends the all-gather's CB, kernels, semaphores and fabric mux kernels to `desc`. Tensor addresses are buffer
// bindings; the GlobalSemaphore addresses are not in the program hash, so the caller re-applies them on every cache
// hit with apply_all_gather_async_minimal_default_semaphore_args().
AllGatherDescriptorArtifacts build_all_gather_async_minimal_default_program_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
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

// Cache-hit refresh for a Program built from build_all_gather_async_minimal_default_program_descriptor: writes the
// current barrier / semaphore addresses into the reader's and writer's common runtime args.
void apply_all_gather_async_minimal_default_semaphore_args(
    tt::tt_metal::Program& program,
    const AllGatherDescriptorArtifacts& artifacts,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore);

}  // namespace ttnn
