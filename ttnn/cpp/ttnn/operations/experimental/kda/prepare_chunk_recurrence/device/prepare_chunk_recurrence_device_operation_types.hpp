// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// How g carries the log decay. PerChannel: one BF16 value per (token, V head, key channel), flat [1, T, HV*K].
// Scalar (GDN): one FP32 value per (V head, token), laid out like beta [HV, N, 32, 1]; the pairwise decay is formed in
// difference form, so it is exact at any decay. Selected by g's shape.
enum class PrepareChunkRecurrenceDecayMode : uint8_t { PerChannel, Scalar };

struct PrepareChunkRecurrenceParams {
    uint32_t sequence_parallel_axis;
    // Part of program identity (different reader and compute paths).
    PrepareChunkRecurrenceDecayMode decay_mode = PrepareChunkRecurrenceDecayMode::PerChannel;
    uint32_t num_heads;
    // Q/K heads; V head hv reads K head hv / (num_heads / num_key_heads). Part of program identity.
    uint32_t num_key_heads;
    uint32_t num_chunks;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t output_bf16_mask = 0;
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

}  // namespace ttnn::experimental::prim
