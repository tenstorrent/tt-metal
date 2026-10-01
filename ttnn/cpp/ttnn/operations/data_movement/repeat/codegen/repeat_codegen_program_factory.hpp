// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Pages a reader/writer moves per turn of its loop.
inline constexpr uint32_t kRepeatBatch = 4;

// The writers wait for the next batch while the previous one is still in flight, so a CB must hold
// two batches.
inline constexpr uint32_t kRepeatCbDepth = 2 * kRepeatBatch;

// A row-major CB slot is one stick, so its footprint grows with the tensor's width. `slot_bytes` is
// the slot, `depth` the slots the CB holds and `batch` the pages a reader/writer moves per turn.
struct RepeatRmCbPlan {
    uint32_t slot_bytes = 0;
    uint32_t batch = 0;
    uint32_t depth = 0;
};

// The largest plan `l1_budget` admits: up to kRepeatBatch pages per turn with two turns in flight, or,
// when only one slot fits, a single slot that the reader and writer take turns on.
inline std::optional<RepeatRmCbPlan> plan_rm_cb(uint32_t slot_bytes, uint64_t l1_budget) {
    const uint64_t max_slots = slot_bytes == 0 ? 0 : l1_budget / slot_bytes;
    if (max_slots == 0) {
        return std::nullopt;
    }
    if (max_slots == 1) {
        return RepeatRmCbPlan{.slot_bytes = slot_bytes, .batch = 1, .depth = 1};
    }
    const auto batch = static_cast<uint32_t>(std::min<uint64_t>(kRepeatBatch, max_slots / 2));
    return RepeatRmCbPlan{.slot_bytes = slot_bytes, .batch = batch, .depth = 2 * batch};
}

// The routing gate sends a row-major leg to codegen only if two slots fit the static L1 window, so
// that on an idle device it double-buffers. The single-slot plan exists for dispatch under L1
// pressure, where it runs slower instead of failing.
inline bool rm_slot_routable(uint64_t slot_bytes, uint64_t l1_budget) {
    return slot_bytes != 0 && 2 * slot_bytes <= l1_budget;
}

// A ROW_MAJOR CB slot: one stick, holding whichever of the leg's input and output aligned pages is
// larger, since DRAM and L1 align differently. The routing gate, the program-cache key and the
// factory all size the slot here; only where the pages come from differs.
inline uint32_t rm_slot_bytes(uint32_t in_aligned_page, uint32_t out_aligned_page) {
    return std::max(in_aligned_page, out_aligned_page);
}

// The Buffer::aligned_page_size() a buffer allocated from `spec` on `device_tensor`'s device will
// have. For buffers that do not exist yet: the gate's leg intermediates, and the op's own output when
// the program-cache key is computed, which never sees the output tensor. Once a buffer exists, read
// its aligned_page_size() instead.
uint32_t spec_aligned_page_bytes(const Tensor& device_tensor, const tt::tt_metal::TensorSpec& spec);

struct RepeatCodegenParams {
    uint32_t rep_dim{};
    uint32_t num_repeats{};
    uint32_t lower_pages{};
    uint32_t rep_dim_pages{};
    uint32_t total_out_pages{};
    // RM only; unused on the TILE branch (tile size is fixed by dtype).
    uint32_t stick_size{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

// The page map a repeat of `input` along `rep_dim` by `num_repeats` addresses: the lower_pages,
// rep_dim_pages, total_out_pages and stick_size fields of RepeatCodegenParams. `input` must be 4D,
// rep_dim in [0, 3]. The router builds the params from it and the prim checks them against it, since
// the kernels address output pages [0, total_out_pages) with no bound of their own.
struct RepeatPageMap {
    uint32_t lower_pages = 0;
    uint32_t rep_dim_pages = 0;
    uint32_t total_out_pages = 0;
    uint32_t stick_size = 0;
};

RepeatPageMap derive_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats);

// How a TILE leg spreads its pages over workers: `cores_in_order[i]` takes the next `work[i]` pages,
// input pages on the direct outer-axis kernel and output pages on the sequenced pair. The routing gate
// replays it to see which shard cores each worker reads, so the gate and the factory both take it
// from here.
struct TileLegSplit {
    bool direct_outer_tile = false;
    tt::tt_metal::CoreRangeSet all_cores;
    std::vector<tt::tt_metal::CoreCoord> cores_in_order;
    std::vector<uint32_t> work;
};

// `params.output_mem_config` must be the placement the leg's output is allocated with.
TileLegSplit plan_tile_leg_split(const Tensor& input, const RepeatCodegenParams& params);

struct RepeatCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

// The ROW_MAJOR CB plan for a leg from `input` to an output of `output_spec`, sized to the L1 free now.
// The program-cache key and the factory both call it after the op's output is allocated, so both see
// the same frontier and the key carries exactly the plan the factory builds.
std::optional<RepeatRmCbPlan> rm_cb_plan_for_call(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

struct RepeatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
