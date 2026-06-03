// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::experimental::prim {

struct AllGatherFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
        tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
        tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
        std::optional<tt::tt_metal::GlobalSemaphore> init_barrier_sem;
        std::optional<tt::tt_metal::GlobalSemaphore> final_barrier_sem;
        uint32_t ring_index = 0;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherParams& operation_attributes,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const Tensor& input,
        const Tensor& output_tensor,
        const std::optional<tt::tt_metal::GlobalSemaphore>& init_barrier_sem,
        const std::optional<tt::tt_metal::GlobalSemaphore>& final_barrier_sem);
};

}  // namespace ttnn::experimental::prim
