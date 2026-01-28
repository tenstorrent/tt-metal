// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        StridedAllGatherAsyncProgramFactory::shared_variables_t ag_shared_variables;
        MinimalMatmulProgramFactory::shared_variables_t mm_shared_variables;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);
};

}  // namespace ttnn::experimental::prim
