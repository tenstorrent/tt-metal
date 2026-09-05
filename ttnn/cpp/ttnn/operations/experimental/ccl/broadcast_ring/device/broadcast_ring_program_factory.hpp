// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <vector>

namespace ttnn::prim {

struct BroadcastRingProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> worker_cores;
        tt::tt_metal::KernelHandle relay_kernel_id{};
        tt::tt_metal::GlobalSemaphore recv_semaphore;  // upstream -> this device, per-chunk data-ready credits
        tt::tt_metal::GlobalSemaphore barrier_semaphore;
        std::vector<tt::tt_metal::GlobalSemaphore> extra_semaphores;  // L1-relay cred sems, kept alive here
        uint32_t ring_index = 0;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const BroadcastRingParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const BroadcastRingInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const BroadcastRingParams& operation_attributes,
        const BroadcastRingInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const BroadcastRingParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const BroadcastRingInputs& tensor_args,
        Tensor& tensor_return_value,
        const tt::tt_metal::GlobalSemaphore& recv_semaphore,
        const tt::tt_metal::GlobalSemaphore& barrier_semaphore);

    // L1-relay variant of create_at: forwards each chunk into the downstream's L1 recv buffer (no per-hop
    // DRAM read), bounded by the backward credit sems. Selected when operation_attributes.use_l1_relay.
    static cached_program_t create_at_l1(
        const BroadcastRingParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const BroadcastRingInputs& tensor_args,
        Tensor& tensor_return_value,
        const tt::tt_metal::GlobalSemaphore& recv_semaphore,
        const tt::tt_metal::GlobalSemaphore& cred_fwd_semaphore,
        const tt::tt_metal::GlobalSemaphore& cred_bwd_semaphore);
};

}  // namespace ttnn::prim
