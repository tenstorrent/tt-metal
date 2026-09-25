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

#include <map>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

struct CombineFabric2dProgramFactory {
    // One ProgramDescriptor per mesh coordinate: each chip sends to its own neighbor, so compile-time args
    // are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// What a program sharing the chip with combine hands it instead of letting it allocate. The routed expert's L1
// arena covers every worker core and must be allocated AFTER fwd_arrived, which is the one allocation combine
// makes: the allocator reserves an address on every core at once, so an arena taken first leaves no room, and
// combine's static circular buffers would clash with it wherever they sit.
struct CombineL1 {
    const tt::tt_metal::GlobalSemaphore* fwd_arrived = nullptr;  // null: combine allocates its own
    tt::tt_metal::Buffer* arena = nullptr;  // null: circular buffers are static, at the allocator base
};

// Where a chip's routed-expert writers report: its collector's core and the per-step count array on it.
struct CollectorTarget {
    tt::tt_metal::CoreCoord worker_virtual;
    uint32_t counts_addr = 0;
};

// The workload CombineFabric2dProgramFactory builds, laid out over `l1`, with each chip's collector recorded in
// `collectors` when the op waits for the routed expert.
tt::tt_metal::WorkloadDescriptor create_combine_workload(
    const CombineFabric2dParams& operation_attributes,
    const CombineFabric2dInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const CombineL1& l1,
    std::map<ttnn::MeshCoordinate, CollectorTarget>* collectors);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
