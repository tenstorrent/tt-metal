// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

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

        // CB handle for tensor backed indices tensor
        tt::tt_metal::CBHandle indices_cb_handle;

        // CB handle for tensor backed scores tensor
        tt::tt_metal::CBHandle scores_cb_handle;

        // CB handle for shared global sharded tensor
        tt::tt_metal::CBHandle sharded_output_cb_handle;

        // CB handle for matmul output
        tt::tt_metal::CBHandle matmul_writer_cb_handle;

        // Combine kernel handles (empty in ComputeOnly mode)
        std::vector<tt::tt_metal::KernelHandle> combine_kernel_handles;

        // CB handle for combine global sharded input tensor (default-constructed in ComputeOnly)
        tt::tt_metal::CBHandle combine_data_cb_handle;

        // CB handle for token counts per expert
        tt::tt_metal::CBHandle expert_tokens_cb_handle;

        // Combine cores (empty in ComputeOnly mode)
        std::vector<CoreCoord> combine_cores;

        // Combine global semaphores (empty in ComputeOnly mode)
        std::vector<GlobalSemaphore> combine_global_semaphores;

        // Path used to build this workload (Full = combine kernels built; ComputeOnly = bypassed).
        MoEComputePath path = MoEComputePath::Full;
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
        const std::optional<GlobalSemaphore>& init_barrier_semaphore,
        const std::optional<GlobalSemaphore>& final_barrier_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MoEComputeParams& args,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores);

ttnn::CoreRange get_moe_worker_mcast_bounding_box(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const uint32_t bh_ring_size = 12);

}  // namespace ttnn::experimental::prim
