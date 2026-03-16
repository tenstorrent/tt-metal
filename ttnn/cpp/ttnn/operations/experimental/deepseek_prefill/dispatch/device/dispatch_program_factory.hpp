// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "dispatch_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

struct DispatchSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores = {};
    GlobalSemaphore init_semaphore;          // Initialized in create_at()
    GlobalSemaphore cross_device_semaphore;  // Initialized in create_at()
};

struct DispatchProgramFactory {
    using shared_variables_t = DispatchSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    static cached_mesh_workload_t create_mesh_workload(
        const DispatchParams& operation_attributes,
        const MeshCoordinateRangeSet& tensor_coords,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const DispatchParams& operation_attributes,
        const MeshCoordinate& mesh_coordinate,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const MeshCoordinateRangeSet& tensor_coords,
        const GlobalSemaphore& init_semaphore,
        const GlobalSemaphore& cross_device_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DispatchParams& operation_attributes,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
