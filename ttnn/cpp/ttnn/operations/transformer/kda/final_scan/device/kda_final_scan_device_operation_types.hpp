// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KdaFinalChunkScanParams {
    uint32_t batch_heads;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t value_dim;
    bool identity_initial_state = false;
    bool state_only = false;
    bool summary_pair = false;
    bool output_bf16 = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaFinalChunkScanInputs {
    Tensor v_beta;
    Tensor kd;
    Tensor q_decay;
    Tensor intra;
    Tensor k_dec_t;
    Tensor final_decay;
    Tensor t_inv;
    std::optional<Tensor> initial_state;
    std::optional<Tensor> identity_tile;
};

}  // namespace ttnn::prim
