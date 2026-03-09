// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "selective_reduce_combine_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct SelectiveReduceCombineProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    const GlobalSemaphore init_semaphore;
    const GlobalSemaphore cross_device_semaphore;
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
}  // namespace ttnn::experimental::prim

namespace ttnn {
using SelectiveReduceCombineProgramArtifacts = experimental::prim::SelectiveReduceCombineProgramArtifacts;

// Builder function that creates kernels and returns artifacts
SelectiveReduceCombineProgramArtifacts build_selective_reduce_combine_program_artifacts(
    tt::tt_metal::Program& program,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& output_tensor,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id);

// Runtime argument override function
void selective_reduce_combine_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& output_tensor,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore);

}  // namespace ttnn
