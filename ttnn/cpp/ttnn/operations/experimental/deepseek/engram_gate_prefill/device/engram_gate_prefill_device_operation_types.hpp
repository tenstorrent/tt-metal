// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::deepseek::engram_gate_prefill {

struct operation_attributes_t {
    // RMS epsilon of the checkpoint's Engram gate.
    float eps = 0.0f;
    // Floor of |dot| before the signed square root (``Engram.clamp_value``).
    float clamp_value = 1e-6f;

    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    // [1, hc, T, D] bf16 TILE, DRAM interleaved: the residual streams the lookup is written into.
    const Tensor& streams;
    // [1, 1, T, (hc + 1) * D] bf16 TILE, DRAM interleaved: the wkv projection, key then value.
    const Tensor& kv;
    // [1, hc, TILE_HEIGHT, D] fp32 TILE, DRAM interleaved: page h is q_weight * k_weight / sqrt(D) of copy h,
    // repeated down the tile's rows so it is a plain tile operand against a 32-token block.
    const Tensor& gate_weight;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::engram_gate_prefill
