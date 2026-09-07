// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

enum class RecurrentChunkScanMode : uint8_t { RECURRENT, SUMMARY };

// Where a chip's causal stream restarts, expressed against its folded layout.
//
// Groups are folded into the leading dimension, so the device cannot recover a
// group index from `batch_heads` alone -- it needs `groups_per_head`. `wrap_chunk`
// is the absolute local chunk at which the stream restarts, 0 meaning none. At
// most one group can contain the wrap, and when `wrap_offset` is zero none does.
struct WrapLayout {
    uint32_t groups = 1;       // G
    uint32_t wrap_group = 0;   // gw = w / g
    uint32_t wrap_offset = 0;  // r  = w % g, 0 == no group straddles
    uint32_t slots = 1;        // S  = G + (r != 0)
    uint32_t real_heads = 1;   // BH = batch_heads / G

    // Slot a group's own chunk-0 seed lives in. The straddling group's pre-wrap
    // half owns slot gw, so every later group shifts up by one.
    uint32_t slot_of(uint32_t group) const { return group + ((wrap_offset != 0 && group > wrap_group) ? 1u : 0u); }
};

struct RecurrentChunkScanParams {
    uint32_t batch_heads;
    uint32_t num_chunks;
    uint32_t key_dim;
    uint32_t value_dim;
    // Groups folded into batch_heads; 1 means ungrouped.
    uint32_t groups_per_head;
    // Local chunk at which the causal stream restarts; 0 means no wrap.
    uint32_t wrap_chunk;
    RecurrentChunkScanMode mode;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct RecurrentChunkScanInputs {
    Tensor v_beta;
    Tensor kd;
    Tensor q_decay;
    Tensor intra;
    Tensor k_dec_t;
    Tensor final_decay;
    Tensor t_inv;
    std::optional<Tensor> initial_state;
    // Seed for the post-wrap loop on the boundary chip: the prefix's final carry,
    // already replicated across SP. Absent when there is no wrap.
    std::optional<Tensor> tail_state;
};

inline WrapLayout wrap_layout(const RecurrentChunkScanParams& attrs) {
    const uint32_t groups = attrs.groups_per_head == 0 ? 1 : attrs.groups_per_head;
    WrapLayout layout;
    layout.groups = groups;
    layout.real_heads = attrs.batch_heads / groups;
    layout.wrap_group = attrs.wrap_chunk / attrs.num_chunks;
    layout.wrap_offset = attrs.wrap_chunk % attrs.num_chunks;
    layout.slots = groups + (layout.wrap_offset != 0 ? 1u : 0u);
    return layout;
}

}  // namespace ttnn::experimental::prim
