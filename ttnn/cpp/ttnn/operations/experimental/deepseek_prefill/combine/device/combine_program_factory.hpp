// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "combine_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine {

struct CombineSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle zero_init_kernel_id = 0;
    std::vector<CoreCoord> cores;
    std::vector<CoreCoord> zero_init_cores;
    GlobalSemaphore init_semaphore;       // Initialized in create_at()
    uint32_t zero_init_semaphore_id = 0;  // Local semaphore ID for reader->writer sync
    uint32_t zero_init_barrier_semaphore_id = 0;  // Barrier: writer signals reader after global init
};

struct CombineProgramFactory {
    using shared_variables_t = CombineSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const CombineParams& operation_attributes,
        const MeshCoordinateRangeSet& tensor_coords,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const CombineParams& operation_attributes,
        const MeshCoordinate& mesh_coordinate,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const MeshCoordinateRangeSet& tensor_coords,
        const GlobalSemaphore& init_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const CombineParams& operation_attributes,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
