// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/runtime_args_data.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "dit_fused_distributed_rmsnorm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct DitFusedDistributedRmsnormSharedVariables {
    // Kernel-owned objects remain stable across Program moves. Dispatch assembly
    // redirects their data(), so cache these objects rather than raw payload pointers.
    tt::tt_metal::RuntimeArgsData* reader_common_args = nullptr;
    tt::tt_metal::RuntimeArgsData* writer_common_args = nullptr;
    // Empty on the local-normalization path; each forwarder uses slots 0 and 1
    // for the stats scratch and ping-pong semaphore addresses.
    std::vector<tt::tt_metal::RuntimeArgsData*> forwarder_runtime_args;
};

struct DitFusedDistributedRmsnormMeshWorkloadFactory {
    using shared_variables_t = DitFusedDistributedRmsnormSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const DitFusedDistributedRmsnormParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DitFusedDistributedRmsnormParams& operation_attributes,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const DitFusedDistributedRmsnormParams& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
