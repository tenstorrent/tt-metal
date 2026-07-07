// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GatedDeltaAttnPreprocessParams {
    uint32_t num_heads;   // BH
    uint32_t num_chunks;  // L / C
    uint32_t chunk_size;  // currently hardwired to 128
    uint32_t key_dim;     // currently hardwired to 128
    uint32_t val_dim;     // currently hardwired to 128
    float diag_alpha;
    bool bf16_value_path;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GatedDeltaAttnPreprocessInputs {
    Tensor q;     // [BH, L, Dk]
    Tensor k;     // [BH, L, Dk]
    Tensor v;     // [BH, L, Dv]
    Tensor beta;  // [BH, L, 1]
    Tensor g;     // [BH, L, 1]
    Tensor triu_ones;
    Tensor tril_mask;
    Tensor eye;
    Tensor lower_causal;
    Tensor eye_32;
};

}  // namespace ttnn::prim
