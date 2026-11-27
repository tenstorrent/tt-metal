// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

// One ProgramDescriptor per mesh coordinate (line index, neighbours and fabric connections are per device).
struct LineReduceScatterProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Called on every cache hit after the buffer bindings are applied. Re-applies the hash-excluded caller-owned
    // GlobalSemaphores (see ReduceScatterMinimalAsyncParams) and re-writes the tensor slots of the common args by role.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const ReduceScatterMinimalAsyncParams& operation_attributes,
        const ReduceScatterMinimalAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

ReduceScatterDescriptorArtifacts build_line_reduce_scatter_minimal_async_program_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    // Unused by Line (the contiguous penult intermediate path is Ring-only); present only so this builder shares
    // a call signature with build_ring_reduce_scatter_minimal_async_program_descriptor, which lets
    // callers pick between them via a single function-pointer/ternary.
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

}  // namespace ttnn::experimental::prim
