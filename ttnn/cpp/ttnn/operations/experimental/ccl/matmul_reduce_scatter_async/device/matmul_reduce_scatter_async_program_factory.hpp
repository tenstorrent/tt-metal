// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MatmulReduceScatterAsyncSharedVariables {
    ttnn::experimental::prim::ReduceScatterProgramArtifacts reduce_scatter_artifacts;
    ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t matmul_shared_variables;
};

struct MatmulReduceScatterAsyncProgramFactory {
    using shared_variables_t = MatmulReduceScatterAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulReduceScatterAsyncParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MatmulReduceScatterAsyncParams& args,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const MatmulReduceScatterAsyncParams& args,
        const ttnn::MeshCoordinate& mesh_coord,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors);
};

}  // namespace ttnn::experimental::prim
