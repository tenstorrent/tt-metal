// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_compute_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeMeshWorkloadFactory {
    struct shared_variables_t {
        // Tilize kernel handles
        std::vector<tt::tt_metal::KernelHandle> tilize_kernel_handles;

        // Tilize cores
        std::vector<CoreCoord> tilize_cores;

        // Matmul kernel handles
        std::vector<tt::tt_metal::KernelHandle> matmul_kernel_handles;

        // Matmul cores
        std::vector<CoreCoord> matmul_cores;

        // CB handle for shared global sharded tensor
        tt::tt_metal::CBHandle sharded_output_cb_handle;

        // CB handle for matmul output
        tt::tt_metal::CBHandle matmul_writer_cb_handle;

        // Combine kernel handles
        std::vector<tt::tt_metal::KernelHandle> combine_kernel_handles;

        // Combine cores
        std::vector<CoreCoord> combine_cores;

        // Combine global semaphores
        std::vector<GlobalSemaphore> combine_global_semaphores;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoEComputeParams& args,
        const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MoEComputeParams& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
        const GlobalSemaphore& init_barrier_semaphore,
        const GlobalSemaphore& final_barrier_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MoEComputeParams& args,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

std::vector<ttnn::CoreCoord> get_moe_combine_cores(ttnn::MeshDevice* mesh_device);

}  // namespace ttnn::experimental::prim
