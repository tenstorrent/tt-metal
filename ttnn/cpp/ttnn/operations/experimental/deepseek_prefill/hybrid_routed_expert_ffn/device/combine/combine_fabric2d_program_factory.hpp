// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <map>
#include <vector>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

// The overlapped fork shares combine_fabric2d's parameter and input structs rather than redeclaring them:
// both ops link into one binary, so a second definition differing by a field would be an ODR violation
// rather than a build error. The overlap-only fields live in that one definition, defaulted.
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dInputs;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dParams;

struct CombineFabric2dProgramFactory {
    // One ProgramDescriptor per mesh coordinate: each chip sends to its own neighbor, so compile-time args
    // are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// What a program sharing the chip with combine hands it instead of letting it allocate. The overlapped routed
// expert passes its caller's fwd_arrived and final_arrived, which outlive every launch, and its per-call L1 arena,
// which covers every worker core: combine places its rings and control region inside that arena rather than at the
// allocator base, where the arena would clash with combine's static circular buffers.
struct CombineL1 {
    // Both or neither; null: combine allocates its own.
    const tt::tt_metal::GlobalSemaphore* fwd_arrived = nullptr;
    const tt::tt_metal::GlobalSemaphore* final_arrived = nullptr;
    // The channel counters, channel_semaphore_count of them. Null: combine allocates its own, which only an
    // op that leaves L1 free may ask for -- see channel_semaphore_count.
    const std::vector<tt::tt_metal::GlobalSemaphore>* channels = nullptr;
    tt::tt_metal::Buffer* arena = nullptr;  // null: circular buffers are static, at the allocator base
    // DRAM, of forwarding_buffer_spec. Null: combine allocates its own at cache miss and the cached workload keeps
    // it, which pins it wherever DRAM was free in the middle of that first forward.
    tt::tt_metal::Buffer* forwarding = nullptr;
};

// The spec of combine's forwarding buffer: one page per relayed token, the token plus its forwarding metadata.
tt::tt_metal::TensorSpec forwarding_buffer_spec(const CombineFabric2dParams& args, uint32_t token_size_bytes);

// How many GlobalSemaphores CombineL1::channels holds for a ring of `ring_extent` chips.
//
// Caller-owned rather than allocated here, for two reasons that each suffice. A neighbouring chip bumps
// vc_arrived and vc_freed across launches, so they must outlive the program exactly as fwd_arrived does.
// And the routed expert's L1 arena takes every free byte before this op is built: a semaphore allocated at
// cache miss would have nowhere to go, and one allocated in the arena's shadow would shrink the NEXT call's
// arena, miss the cache, and allocate again.
//
// There are too many of them to be program semaphores. Of the 16 ids a core has, the ring and untilizer
// handshakes already take 11, and a ring of 8 wants 19 counters.
//
// The caller hands over that many and nothing else: which one is which is this file's business, and it reads
// them in order as credit_done, then each channel's arrived, freed and outbox.
uint32_t channel_semaphore_count(uint32_t ring_extent);

// Where a chip's routed-expert writers report: its collector's core and the per-step count array on it.
struct CollectorTarget {
    tt::tt_metal::CoreCoord logical;
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
