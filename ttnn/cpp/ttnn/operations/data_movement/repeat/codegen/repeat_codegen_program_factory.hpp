// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

inline constexpr uint32_t kRepeatBatch = 4;

// Two batches: the writers wait for the next batch while the previous one is still in flight.
inline constexpr uint32_t kRepeatCbDepth = 2 * kRepeatBatch;

struct RepeatRmCbPlan {
    uint32_t slot_bytes = 0;
    uint32_t batch = 0;
    uint32_t depth = 0;
};

// When only one slot fits, the reader and writer take turns on it.
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

// Routing demands two slots so an idle device double-buffers; the single-slot plan only covers L1 pressure.
inline bool rm_slot_routable(uint64_t slot_bytes, uint64_t l1_budget) {
    return slot_bytes != 0 && 2 * slot_bytes <= l1_budget;
}

// DRAM and L1 align differently, so the slot holds the larger aligned page.
inline uint32_t rm_slot_bytes(uint32_t in_aligned_page, uint32_t out_aligned_page) {
    return std::max(in_aligned_page, out_aligned_page);
}

// For buffers not yet allocated; once a buffer exists, read its aligned_page_size() instead.
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

// The prim checks params against this because the kernels address output pages with no bound of their own.
struct RepeatPageMap {
    uint32_t lower_pages = 0;
    uint32_t rep_dim_pages = 0;
    uint32_t total_out_pages = 0;
    uint32_t stick_size = 0;
};

RepeatPageMap derive_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats);

struct RepeatCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

// Sized to the L1 free now; the cache key and the factory must both call it after the output is allocated.
std::optional<RepeatRmCbPlan> rm_cb_plan_for_call(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

struct RepeatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
