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

// Pages a reader/writer moves per turn of its loop.
inline constexpr uint32_t kRepeatBatch = 4;

// The writers wait for the next batch while the previous one is still in flight, so a CB must hold
// two batches.
inline constexpr uint32_t kRepeatCbDepth = 2 * kRepeatBatch;

// A row-major CB slot is one stick, so its footprint grows with the tensor's width. The factory
// shrinks batch and depth together to what the L1 left free admits, and the routing gate rejects a
// stick for which even a one-page batch does not fit. Both call this so they cannot disagree.
struct RepeatRmCbPlan {
    uint32_t batch = 0;
    uint32_t depth = 0;
};

inline std::optional<RepeatRmCbPlan> plan_rm_cb(uint64_t slot_bytes, uint64_t l1_budget) {
    const uint64_t max_slots = slot_bytes == 0 ? 0 : l1_budget / slot_bytes;
    const uint64_t batch = std::min<uint64_t>(kRepeatBatch, max_slots / 2);
    if (batch == 0) {
        return std::nullopt;
    }
    return RepeatRmCbPlan{.batch = static_cast<uint32_t>(batch), .depth = static_cast<uint32_t>(2 * batch)};
}

// L1 per worker core that the allocator can ever hand to buffers and CBs, independent of what is
// allocated right now. The routing gate budgets against this so its answer does not move with the
// live allocator state between routing and a program-cache miss.
uint64_t static_l1_window(const Tensor& input);

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

// The CB plan a ROW_MAJOR leg is built with for a `slot_bytes` slot, sized to the L1 left free right
// now; nullopt for TILE or when not even a one-page batch fits. The program-cache key carries it, so a
// program built under one allocator state is never replayed under a key that implies another.
std::optional<RepeatRmCbPlan> live_rm_cb_plan(const Tensor& input, uint32_t slot_bytes);

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

struct RepeatCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

struct RepeatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
