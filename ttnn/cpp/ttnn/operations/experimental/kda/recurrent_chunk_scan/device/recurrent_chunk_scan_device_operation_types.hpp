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

struct RecurrentChunkScanParams {
    uint32_t batch_heads;
    uint32_t num_chunks;
    uint32_t key_dim;
    uint32_t value_dim;
    // Groups folded into batch_heads; 1 means ungrouped.
    uint32_t groups_per_head;
    RecurrentChunkScanMode mode;
    // SUMMARY only. The output is one [S, BH, K, K+V] tensor of complement-form [E | B] rows (E = A - I): slot 0
    // holds head transforms and, with tail, slot 1 holds tail transforms (S = 2). With init, every slot without a
    // valid segment holds the identity, (0, 0); otherwise such slots are unspecified.
    bool tail = false;
    bool init = true;
    // SUMMARY with one group per head only: also write each chunk's zero-seeded output F(0) and identity-seeded output
    // F(I) as one terms tensor, and the final segment's transform in complement form (E = A - I, B: [BH, K, V] each),
    // all FLOAT32. Once the entry state is known, a chunk's output is F(0) + (F(I) - F(0)) @ entry and the rank's final
    // state is entry + (E @ entry + B).
    // The terms are [BH * blocks, 2, block_chunks * 32, V]: each block of a head's chunks holds its F(0) rows, then
    // its F(I) rows, so a height shard of one block holds both (emit_block_chunks).
    bool emit_outputs = false;
    // Memory of the emitted terms (default: output_mem_config), e.g. sharded one block per consumer core.
    std::optional<tt::tt_metal::MemoryConfig> emit_mem_config = std::nullopt;
    uint32_t sequence_parallel_axis;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// The chunks of an emitted terms block: one height shard's worth when sharded, else a head's.
inline uint32_t emit_block_chunks(const RecurrentChunkScanParams& attrs) {
    const auto& memory = attrs.emit_mem_config;
    return memory && memory->shard_spec() ? memory->shard_spec()->shape[0] / (2 * 32) : attrs.num_chunks;
}

struct RecurrentChunkScanInputs {
    Tensor v_beta;
    Tensor kd;
    Tensor q_decay;
    Tensor intra;
    Tensor k_dec_t;
    Tensor final_decay;
    Tensor t_inv;
    std::optional<Tensor> group_entry_states;
    // Seed for the post-wrap loop on the first rank: the prefix's final carry,
    // already replicated across SP. Required for SP recurrence.
    std::optional<Tensor> tail_entry_states;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
