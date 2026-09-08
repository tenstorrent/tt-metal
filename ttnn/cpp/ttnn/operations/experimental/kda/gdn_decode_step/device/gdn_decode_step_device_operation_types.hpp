// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepParams {
    uint32_t num_value_heads;
    uint32_t num_key_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnDecodeStepInputs {
    Tensor qkv;     // [1, 1, 2*Nk*Dk + Nv*Dv] bf16, post conv+silu
    Tensor beta;    // [1, 1, Nv] fp32/bf16
    Tensor g;       // [1, 1, Nv] fp32/bf16 (log decay)
    Tensor state;   // [1, Nv, Dk, Dv] fp32, updated in place
    Tensor weight;  // [Dv] bf16 gated-norm weight
};

}  // namespace ttnn::experimental::prim
