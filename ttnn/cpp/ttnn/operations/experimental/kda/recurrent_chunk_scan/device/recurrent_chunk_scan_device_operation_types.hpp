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
};

}  // namespace ttnn::experimental::prim
