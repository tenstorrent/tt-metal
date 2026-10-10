// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::bringup {

struct PrepareChunkRecurrenceParams {
    uint32_t sequence_parallel_axis;
    uint32_t num_heads;
    uint32_t num_chunks;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t output_bf16_mask = 0;
    // Multiplies the per-key log decay before its within-chunk cumulative sum.
    float gate_scale = 1.0F;
    // When set, beta holds pre-sigmoid logits read in place from these columns of a token-major BF16 tensor, and
    // preparation applies the sigmoid.
    std::optional<uint32_t> beta_logits_column_offset;
    // Bring-up option (GLM-5.3): form the anchored gate exponents G - G_last/2 and G_last/2 as exact matmuls of g with
    // constant masks and exponentiate them in DST, instead of the FP32 (TF32-read) subtraction and copies.
    bool precise_gate_factors = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct PrepareChunkRecurrenceInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor g;
    Tensor beta;
    std::optional<Tensor> actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim::bringup
