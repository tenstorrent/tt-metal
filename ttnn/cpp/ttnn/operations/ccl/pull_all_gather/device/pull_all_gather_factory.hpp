// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pull_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::ccl {

struct PullAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::GlobalSemaphore barrier_sem;
        uint32_t device_idx = 0;           // which row block this device owns
        std::vector<uint32_t> route_args;  // num_routes + kMaxRoutes * 7 words, or the peer mask alone
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const PullAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const PullAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const PullAllGatherParams& operation_attributes,
        const PullAllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const PullAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const PullAllGatherInputs& tensor_args,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& barrier_sem);
};

}  // namespace ttnn::operations::ccl
