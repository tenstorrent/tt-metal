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
    uint32_t sequence_parallel_axis;
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
    std::optional<Tensor> group_entry_states;
    // Seed for the post-wrap loop on the first rank: the prefix's final carry,
    // already replicated across SP. Required for SP recurrence.
    std::optional<Tensor> tail_entry_states;
    Tensor actual_start;
};

}  // namespace ttnn::experimental::prim
