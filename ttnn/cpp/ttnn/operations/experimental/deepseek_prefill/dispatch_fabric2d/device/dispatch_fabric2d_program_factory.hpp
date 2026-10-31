// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dispatch_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

struct DispatchFabric2dProgramFactory {
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    // One ProgramDescriptor per mesh coordinate: each chip's senders name their own downstream workers,
    // so compile-time args are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DispatchFabric2dParams& args,
        const DispatchFabric2dInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// The stream cores use a fixed L1 region that the allocator does not own, so every launch, cached or not,
// checks that no L1 allocation made since has reached down into it. `token_bytes` is the output payload's
// aligned page size.
void validate_l1_region(const DispatchFabric2dParams& args, uint32_t token_bytes);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
