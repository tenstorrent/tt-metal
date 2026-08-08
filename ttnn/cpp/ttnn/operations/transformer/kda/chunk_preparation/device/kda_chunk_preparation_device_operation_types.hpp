// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KdaChunkPreparationParams {
    uint32_t batch_heads;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t value_dim;
    bool v_flat = false;
    uint32_t value_heads = 0;
    bool qk_flat = false;
    uint32_t key_heads = 0;
    bool gate_flat = false;
    bool normalize_qk = false;
    float scale = 1.0F;
    uint32_t output_bf16_mask = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaChunkPreparationInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor g;
    Tensor beta;
    Tensor eye;
    Tensor tril;
    Tensor ones;
    Tensor masks;
};

}  // namespace ttnn::prim
