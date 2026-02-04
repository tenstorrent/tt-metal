// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation_types.hpp"

namespace ttnn::prim {
struct AllBroadcastProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
        tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
        tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
        tt::tt_metal::GlobalSemaphore semaphore;
        tt::tt_metal::GlobalSemaphore barrier_semaphore;
        uint32_t ring_index = 0;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllBroadcastParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensor& input,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllBroadcastParams& operation_attributes,
        const Tensor& input,
        std::vector<Tensor>& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const AllBroadcastParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const Tensor& input,
        std::vector<Tensor>& output_tensors,
        const tt::tt_metal::GlobalSemaphore& semaphore,
        const tt::tt_metal::GlobalSemaphore& barrier_semaphore);
};

}  // namespace ttnn::prim
