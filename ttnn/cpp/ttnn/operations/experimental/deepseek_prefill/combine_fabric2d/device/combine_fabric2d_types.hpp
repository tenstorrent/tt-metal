// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// MoE prefill combine over an explicitly-forwarded 2D fabric: expert-processed tokens go back to the chips
// they came from, chip-local DRAM -> eth -> destination chip's DRAM.
//
// PHASE 10 INTERFACE. Up to phase 9 the caller said what to move with an explicit list of movement
// descriptors. It now says it the way the production combine op does: the work is described by control
// tensors already staged in DRAM, and the op discovers it on device.
//
//   dispatched_buffer     the tokens. Page = one token; a chip's page range for one expert holds that
//                         expert's tokens grouped by the chip they ORIGINATED on, which is where they must
//                         now go back to.
//   dispatched_metadata   3 int32 per token — (linearized_coord, token_idx, topk_idx). The token's slot in
//                         the destination's output is page `token_idx * num_experts_per_tok + topk_idx`.
//   expert_token_counts   tokens per expert, summed over all origin chips. Closes the last origin chip's
//                         run, which `expert_offsets` alone cannot.
//   expert_region_offsets where each expert's region starts in `dispatched_buffer`.
//   expert_offsets        the ONE tensor production does not take: where each ORIGIN chip's run starts
//                         inside each expert's region. Production rediscovers a token's destination per
//                         token from the metadata; we want whole runs, so we are handed the boundaries.
//
// The op still owns everything about HOW: which cores run, which cable each one drives, how a destination
// more than one hop away is reached, and how the work is split across producers. None of that is visible
// to the caller.
//
// `device` is FIRST so the framework's get_first_object_of_type<MeshDevice*>() over attribute_values()
// (tuple element 0) finds the mesh.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    // ---- Shape, named exactly as the production op names it.
    uint32_t dispatch_group_size = 8;  // chips per dispatch group = the ring extent along `axis`
    uint32_t experts_per_chip = 2;     // experts hosted on each chip
    uint32_t num_experts_per_tok = 2;  // top-k; the output has this many slots per token
    uint32_t seq_len_per_chip = 640;   // tokens per chip, i.e. the output's token dimension
    uint32_t axis = 0;                 // mesh axis the ring runs along (production's cluster_axis)
    uint32_t num_links = 2;            // cables per neighbour
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    // Output allocation, also as production takes it. `init_zeros` is accepted only as false: slots with no
    // expert contribution are left as-allocated, exactly as the production op leaves them when asked not to
    // zero, and the caller's validation only looks at slots it knows were routed.
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    bool init_zeros = false;
    // ---- Tuning knobs. Ours, not production's: they change how fast the op is, never what it computes.
    // Depth of the L1 ring between the reader and the producer, in tokens. Accuracy holds for any value
    // >= 2. Slots are claimed and released in batches of num_l1_slots / 2.
    uint32_t num_l1_slots = 8;
    // Forwarded tokens between semaphore bumps to the downstream reader. A bump ALWAYS follows a chunk's
    // sentinel regardless, so this only sets how finely the downstream reader can pipeline WITHIN a chunk.
    uint32_t fwd_bump_every = 32;
    // Order in which a reader works through its own assignments and the forwarding chunks it relays.
    //   0 = nearest destination first, then all forwarding chunks.
    //   1 = furthest first with forwarding interleaved, so downstream cores are fed earlier.
    uint32_t assignment_order = 1;
    // Fine-grained stall attribution in the producer. Off to quote a number, on to explain one.
    uint32_t stall_telemetry = 0;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "device",
        "dispatch_group_size",
        "experts_per_chip",
        "num_experts_per_tok",
        "seq_len_per_chip",
        "axis",
        "num_links",
        "topology",
        "output_mem_config",
        "init_zeros",
        "num_l1_slots",
        "fwd_bump_every",
        "assignment_order",
        "stall_telemetry");
    auto attribute_values() const {
        return std::forward_as_tuple(
            device,
            dispatch_group_size,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            axis,
            num_links,
            topology,
            output_mem_config,
            init_zeros,
            num_l1_slots,
            fwd_bump_every,
            assignment_order,
            stall_telemetry);
    }
};

// The five control/data tensors, all caller-staged in DRAM. The first four are exactly production's, in
// production's order; `expert_offsets` is the extra one. The output is NOT here — the op allocates it, as
// production does.
struct CombineFabric2dInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    ttnn::Tensor expert_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
