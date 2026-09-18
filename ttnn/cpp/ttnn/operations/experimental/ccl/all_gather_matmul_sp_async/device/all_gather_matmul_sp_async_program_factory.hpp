// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMatmulSpAsyncSharedVariables {
    ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t matmul;
    AllGatherProgramArtifacts all_gather;
};

struct AllGatherMatmulSpAsyncMeshWorkloadFactory {
    using shared_variables_t = AllGatherMatmulSpAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherMatmulSpAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherMatmulSpAsyncInputs& tensor_args,
        AllGatherMatmulSpAsyncResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherMatmulSpAsyncParams& operation_attributes,
        const AllGatherMatmulSpAsyncInputs& tensor_args,
        AllGatherMatmulSpAsyncResult& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherMatmulSpAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coord,
        const AllGatherMatmulSpAsyncInputs& tensor_args,
        AllGatherMatmulSpAsyncResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
