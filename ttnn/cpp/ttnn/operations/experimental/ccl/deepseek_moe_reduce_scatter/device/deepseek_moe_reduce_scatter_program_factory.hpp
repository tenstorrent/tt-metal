// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepseek_moe_reduce_scatter_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterMeshWorkloadFactory {
    struct shared_variables_t {
        tt::tt_metal::GlobalSemaphore op_semaphore;
        tt::tt_metal::GlobalSemaphore pre_op_barrier_semaphore;
        DeepseekMoEReduceScatterProgramArtifacts program_artifacts;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const tt::tt_metal::GlobalSemaphore& op_semaphore,
        const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
