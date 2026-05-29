// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct StridedAllGatherAsyncProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
        std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
        std::vector<CoreCoord> all_cores;
        uint32_t num_links;
        uint32_t num_directions_per_link;
        uint32_t num_workers_per_direction;
        uint32_t num_mux_cores_per_direction_per_link;
        uint32_t num_cores_per_link;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const StridedAllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const StridedAllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

    static shared_variables_t strided_all_gather_async_minimal_default_helper(
        tt::tt_metal::Program& program,
        const Tensor& input_tensor,
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
        std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
        bool read_local_slice_from_input,
        std::optional<uint32_t> num_workers_per_direction_opt,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<uint32_t> mm_cores_y,
        std::optional<uint32_t> mm_block_ht,
        std::optional<uint32_t> mm_block_wt,
        CoreCoord core_grid_offset = CoreCoord(0, 0));

    // ProgramDescriptor variant of strided_all_gather_async_minimal_default_helper.
    //
    // Appends KernelDescriptors / CBDescriptors / SemaphoreDescriptors to `desc`
    // instead of creating them directly on a Program.  Runtime args for the
    // input/output tensors are emplaced through
    // KernelDescriptor::emplace_runtime_args(Buffer*) so the framework's fast
    // cache-hit path patches their base addresses every dispatch -- no
    // override_runtime_arguments hook is needed for tensor addrs.
    //
    // GlobalSemaphore addresses (in `semaphore`) remain stable for the workload
    // lifetime (parked on workload_descriptor.semaphores by the caller) and are
    // encoded as plain uint32_t.
    //
    // The KernelHandle values inside the returned shared_variables_t are
    // descriptor-kernel-indices (positions in desc.kernels) -- consumers that
    // need to wire up further patching can use them just like Program KernelHandles.
    //
    // Sub-helpers used (all have descriptor overloads on this branch):
    //   - StridedAllGatherFusedOpSignaler::init_all_gather(ProgramDescriptor&, ...)
    static shared_variables_t strided_all_gather_async_minimal_default_helper_descriptor(
        tt::tt_metal::ProgramDescriptor& desc,
        const Tensor& input_tensor,
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
        std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
        bool read_local_slice_from_input,
        std::optional<uint32_t> num_workers_per_direction_opt,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<uint32_t> mm_cores_y,
        std::optional<uint32_t> mm_block_ht,
        std::optional<uint32_t> mm_block_wt,
        CoreCoord core_grid_offset = CoreCoord(0, 0));

    static void override_runtime_arguments_per_program(
        const shared_variables_t& shared_variables,
        tt::tt_metal::Program& program,
        const StridedAllGatherAsyncParams& attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const StridedAllGatherAsyncParams& operation_attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
