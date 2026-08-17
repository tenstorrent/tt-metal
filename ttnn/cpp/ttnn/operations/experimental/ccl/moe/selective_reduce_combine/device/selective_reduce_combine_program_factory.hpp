// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "selective_reduce_combine_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace detail {

struct SelectiveReduceCombineWorkerLayout {
    std::vector<uint32_t> data_parallel_sizes_bytes;
    uint32_t num_data_parallel_cores = 0;
    uint32_t num_worker_cores = 0;
};

SelectiveReduceCombineWorkerLayout compute_worker_layout(
    const Tensor& input_tensor,
    uint32_t hidden_size,
    uint32_t num_token_parallel_cores,
    uint32_t num_data_parallel_cores,
    bool local_combine = false);

}  // namespace detail

struct SelectiveReduceCombineProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::CBHandle data_cb_handle{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    // Owned by the artifacts for the standalone UnifiedSelectReduce op. nullopt for the
    // fused moe_compute FullLocal path (writer compiles out init/final barrier handling),
    // in which case the kernel runtime args carry a placeholder address of 0.
    std::optional<GlobalSemaphore> init_semaphore;
    std::optional<GlobalSemaphore> cross_device_semaphore;
};

struct UnifiedSelectReduce {
    using operation_attributes_t = SelectiveReduceCombineParams;
    using tensor_args_t = SelectiveReduceCombineTensors;
    using tensor_return_value_t = ttnn::Tensor;

    using shared_variables_t = SelectiveReduceCombineProgramArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const GlobalSemaphore& init_semaphore,
        const GlobalSemaphore& cross_device_semaphore);
};

// Builder function that creates kernels and returns artifacts.
// `init_semaphore` / `cross_device_semaphore` are passed as optionals: the builder uses their
// addresses (0 when nullopt) for writer kernel runtime args, and stores them in the returned
// artifacts for ownership. nullopt is used by the fused moe_compute FullLocal path, whose
// writer compiles out all init/final barrier handling.
SelectiveReduceCombineProgramArtifacts build_selective_reduce_combine_program_artifacts(
    tt::tt_metal::Program& program,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_semaphore,
    const std::optional<GlobalSemaphore>& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_cores = 0,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id = std::nullopt);

// Runtime argument override function. Semaphore kernel runtime-arg slots are written as
// raw addresses; pass 0 for the fused moe_compute FullLocal path (unused by the writer).
void selective_reduce_combine_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::CBHandle data_cb_handle,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    uint32_t init_semaphore_addr,
    uint32_t cross_device_semaphore_addr,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore);

}  // namespace ttnn::experimental::prim
