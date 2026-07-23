// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "high_bw_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather {

struct HighBwAllGatherUnicastFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> worker_cores;
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::GlobalSemaphore data_valid_sem;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const HighBwAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const HighBwAllGatherParams& operation_attributes,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const HighBwAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const HighBwAllGatherInputs& tensor_args,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& data_valid_sem);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather
