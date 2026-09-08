// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>

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
    // fused-conv mode: qkv is the full projection row [q|k|v|z|a|b]; conv + gates computed in-kernel
    bool fuse_conv = false;
    uint32_t qkvz_dim = 0;  // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
};

struct GdnDecodeStepInputs {
    Tensor qkv;     // [1, 1, 2*Nk*Dk + Nv*Dv] bf16, post conv+silu
    Tensor beta;    // [1, 1, Nv] fp32/bf16
    Tensor g;       // [1, 1, Nv] fp32/bf16 (log decay)
    Tensor state;   // [1, Nv, Dk, Dv] fp32, updated in place
    Tensor weight;  // [Dv] bf16 gated-norm weight
    std::optional<Tensor>
        conv_hist;  // fused-conv mode: packed history [Nv, 4, 32, 32] bf16 (slot 3 newest), shifted in place
    std::optional<Tensor> conv_taps;  // fused-conv mode: packed taps [Nv, 4, 32, 32] bf16 (row c = channel chunk c)
};

}  // namespace ttnn::experimental::prim
