// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::program {

struct StridedAllGatherMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        strided_all_gather_async::program::StridedAllGatherAsyncProgramFactory::shared_variables_t ag_shared_variables;
        minimal_matmul::program::MinimalMatmulProgramFactory::shared_variables_t mm_shared_variables;
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
        tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::program
