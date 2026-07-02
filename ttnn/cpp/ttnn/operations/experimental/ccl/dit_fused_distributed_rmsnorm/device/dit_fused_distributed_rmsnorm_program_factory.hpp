// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "dit_fused_distributed_rmsnorm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct DitFusedDistributedRmsnormSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    // Forwarder kernels (AG path) + their cores; used to refresh the stats DRAM
    // scratch address on cache hits. Empty on the is_tp_1 path.
    std::vector<tt::tt_metal::KernelHandle> forwarder_kernel_ids;
    std::vector<tt::tt_metal::CoreCoord> forwarder_cores;
    std::vector<tt::tt_metal::CoreCoord> cores;
    // Index of the stats DRAM scratch address inside the worker-writer's
    // runtime-args vector. Set on the all-gather path (TP>1, whole-row norm);
    // empty on the is_tp_1 path. The address changes per launch because the
    // scratch is a regular device tensor allocated by create_output_tensors, so
    // override_runtime_arguments refreshes this slot on cache hits.
    std::optional<size_t> stats_dram_addr_writer_arg_idx;
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
        const DitFusedDistributedRmsnormParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
