// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// One work unit assembles a fixed-size slab of one output page. This is the transport's only
// per-core lever: pushing it up trades a wider L1 scratch region for fewer, larger NoC bursts;
// this value keeps every burst comfortably under the architecture's NoC max transaction size
// on both currently-supported alignments (16B/32B/64B page pitches).
inline constexpr uint32_t kReshapeRmArbitrarySlabBytes = 4096;

// Host-side transport geometry for the fixed-slab arbitrary-width RM reshape. ROW_MAJOR tensors
// are physically a sequence of aligned pages ("sticks"), but reshape is defined over the tightly
// packed logical byte stream; each output page is cut into slab-sized slabs, and each slab is
// projected onto the packed byte stream to find which source page(s) it reads from. This mirrors
// the reader/writer kernels' own recomputation of that projection (they take only a linear
// work-unit index and derive everything else from it), so the two must stay in lock-step. Shared
// between the program factory (which builds the descriptor from it) and the correctness gate
// (which must reject whatever the factory's plan cannot fit in L1), so they cannot drift.
struct ReshapeRmArbitraryPlan {
    uint32_t old_stick_bytes = 0;
    uint32_t old_page_bytes = 0;
    uint32_t new_stick_bytes = 0;
    uint32_t new_page_bytes = 0;
    uint32_t input_alignment = 0;
    uint32_t output_alignment = 0;
    uint32_t slab_bytes = 0;
    uint32_t slab_slot_bytes = 0;
    uint32_t slabs_per_output = 0;
    uint32_t total_units = 0;
    uint32_t noc_max_burst_bytes = 0;
    // Per-unit scratch region: holds every aligned source window contributing to one slab. A slab
    // spans at most ceil(slab/old_stick)+1 source sticks, each aligned read overshoots by less than
    // 2*input_alignment on either side, plus one alignment unit for the scratch base's own align-up.
    uint32_t region_stride = 0;
    // 0 when even the minimum (single-unit-per-barrier) staging area does not fit in L1 -- the
    // caller must treat that as "this shape cannot be served" rather than build an oversized CB.
    uint32_t nabatch = 0;
};

inline uint32_t reshape_rm_align_down(uint32_t value, uint32_t alignment) { return value / alignment * alignment; }
inline uint32_t reshape_rm_align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline ReshapeRmArbitraryPlan plan_reshape_rm_arbitrary_transport(
    uint32_t old_stick_bytes,
    uint32_t new_stick_bytes,
    uint32_t num_new_sticks,
    uint32_t input_alignment,
    uint32_t output_alignment,
    uint32_t noc_max_burst_bytes,
    uint32_t usable_l1_bytes) {
    ReshapeRmArbitraryPlan plan;
    plan.old_stick_bytes = old_stick_bytes;
    plan.new_stick_bytes = new_stick_bytes;
    plan.input_alignment = input_alignment;
    plan.output_alignment = output_alignment;
    plan.noc_max_burst_bytes = noc_max_burst_bytes;

    // One source read can carry alignment padding on both sides; leave that headroom below the
    // architecture's maximum NoC transaction size.
    const uint32_t payload_cap = noc_max_burst_bytes - 2 * input_alignment;
    const uint32_t slab_cap = std::min<uint32_t>(kReshapeRmArbitrarySlabBytes, payload_cap);
    plan.slab_bytes = reshape_rm_align_down(slab_cap, output_alignment);
    if (plan.slab_bytes == 0) {
        return plan;
    }

    plan.old_page_bytes = reshape_rm_align_up(old_stick_bytes, input_alignment);
    plan.new_page_bytes = reshape_rm_align_up(new_stick_bytes, output_alignment);
    plan.slabs_per_output = (new_stick_bytes + plan.slab_bytes - 1) / plan.slab_bytes;
    plan.total_units = num_new_sticks * plan.slabs_per_output;

    // The final output slab may additionally carry one alignment unit of page padding; the slot
    // has room for an aligned staging pointer plus an aligned source window whose head and tail
    // both surround the logical payload.
    plan.slab_slot_bytes = reshape_rm_align_up(plan.slab_bytes + output_alignment - 1, 16);

    const uint32_t max_windows = plan.slab_bytes / std::max<uint32_t>(1, plan.old_stick_bytes) + 2;
    const uint32_t region = plan.slab_bytes + max_windows * 2 * input_alignment;
    plan.region_stride = reshape_rm_align_up(region, input_alignment);

    // Batch as many units per NoC read barrier as L1 allows (fewer barriers -> less dispatch
    // overhead); tiers mirror the reader/writer's shared batching discipline. nabatch stays 0
    // (unservable) when even a single unit's staging area does not fit.
    for (uint32_t candidate : {8u, 4u, 2u, 1u}) {
        const uint64_t scratch_bytes = static_cast<uint64_t>(candidate) * plan.region_stride;
        const uint64_t out_bytes = static_cast<uint64_t>(2) * candidate * plan.slab_slot_bytes;
        if (scratch_bytes + out_bytes <= usable_l1_bytes) {
            plan.nabatch = candidate;
            break;
        }
    }
    return plan;
}

struct ReshapeCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeCodegenParams& operation_attributes,
        const ReshapeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
