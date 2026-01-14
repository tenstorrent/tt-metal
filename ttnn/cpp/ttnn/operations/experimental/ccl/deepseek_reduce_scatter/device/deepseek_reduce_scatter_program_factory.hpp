// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepseek_reduce_scatter_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail {

struct DeepseekReduceScatterMeshWorkloadFactory {
    struct shared_variables_t {
        tt::tt_metal::GlobalSemaphore op_semaphore;
        std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores;
        DeepseekReduceScatterProgramArtifacts program_artifacts;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const tt::tt_metal::GlobalSemaphore& op_semaphore,
        const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// Builder function for ring topology - creates program artifacts
DeepseekReduceScatterProgramArtifacts build_deepseek_reduce_scatter_program_artifacts(
    tt::tt_metal::Program& program,
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores,
    uint32_t num_links);

// Override runtime arguments helper for ring topology
void deepseek_reduce_scatter_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_directions_per_link,
    const std::vector<tt::tt_metal::CBHandle>& input_cb_handles,
    const std::vector<tt::tt_metal::CBHandle>& intermediate_cb_handles,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores,
    uint32_t num_links,
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor);

}  // namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail
