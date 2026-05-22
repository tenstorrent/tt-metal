// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "wan_fused_distributed_rmsnorm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct WanFusedDistributedRmsnormSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    std::vector<tt::tt_metal::CoreCoord> cores;
    // Persistent DRAM scratch for stats AG (TP>1 only). Holds it alive for
    // the lifetime of the cached program so the address remains valid across
    // trace replays. Allocated via tt_metal::CreateBuffer in create_at.
    std::shared_ptr<tt::tt_metal::Buffer> stats_dram_buffer;
};

struct WanFusedDistributedRmsnormMeshWorkloadFactory {
    using shared_variables_t = WanFusedDistributedRmsnormSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const WanFusedDistributedRmsnormParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const WanFusedDistributedRmsnormInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const WanFusedDistributedRmsnormParams& operation_attributes,
        const WanFusedDistributedRmsnormInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const WanFusedDistributedRmsnormParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const WanFusedDistributedRmsnormInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
