// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_ce_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherCeDefaultMeshWorkloadFactory {
    using shared_variables_t = AllGatherProgramArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherCeParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherCeInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherCeParams& operation_attributes,
        const AllGatherCeInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherCeParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherCeInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
