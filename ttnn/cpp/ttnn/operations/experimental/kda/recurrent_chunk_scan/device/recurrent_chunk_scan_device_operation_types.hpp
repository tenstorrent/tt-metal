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
    // Local chunk at which the causal stream restarts; 0 means no wrap.
    uint32_t wrap_chunk;
    // Half-open chunk range this pass consumes, so a caller can summarize a
    // sub-range without slicing the prepared terms. chunk_count 0 means "to the
    // end". Both uniform across the mesh.
    uint32_t chunk_start;
    uint32_t chunk_count;
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

}  // namespace ttnn::experimental::prim
