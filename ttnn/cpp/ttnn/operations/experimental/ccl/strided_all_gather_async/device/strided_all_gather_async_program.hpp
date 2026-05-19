// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct StridedAllGatherAsyncProgramFactory {
    // Retained for strided_all_gather_minimal_matmul_async, which still drives a
    // legacy Program& factory and needs to splice the strided-all-gather body
    // into its own program.  Used by override_runtime_arguments_per_program.
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

    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index / forward & backward neighbours
    // vary across the mesh).  GlobalSemaphores live on
    // StridedAllGatherAsyncParams (caller-allocated) — no workload-scoped
    // resources needed.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const StridedAllGatherAsyncParams& operation_attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Legacy Program& helper retained for
    // strided_all_gather_minimal_matmul_async (a separate op that fuses an
    // all-gather body with a matmul body into a single Program).  When that
    // op migrates to contract 2, this helper can be removed.
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

    static void override_runtime_arguments_per_program(
        const shared_variables_t& shared_variables,
        tt::tt_metal::Program& program,
        const StridedAllGatherAsyncParams& attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
