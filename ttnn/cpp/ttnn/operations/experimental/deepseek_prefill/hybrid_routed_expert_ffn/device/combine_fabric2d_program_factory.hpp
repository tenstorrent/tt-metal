// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "combine_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

struct CombineFabric2dProgramFactory {
    // One ProgramDescriptor per mesh coordinate: each chip sends to its own neighbor, so compile-time args
    // are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        // The shared L1 arena. The untilizer's circular buffers are bound into it rather than
        // statically placed, so the two halves lay memory into one allocation instead of both
        // claiming the allocator base.
        tt::tt_metal::Buffer* l1_arena,
        uint32_t arena_bytes_per_core);
};

// Bytes above the allocator base that combine hand-places on its stream-worker cores. Those cores
// carry no circular buffers, so an arena descending into them would go unnoticed; size it to start
// above this.
uint32_t stream_worker_l1_bytes(const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
