// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

struct FusedQKVSDPAParams {
    // Query heads in the fused tensor. The head axis is not in the shape, so it cannot be inferred:
    // [B, 1, S, 3*num_heads*head_dim] is all the tensor knows.
    uint32_t num_heads = 0;
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct FusedQKVSDPAInputs {
    // q|k|v concatenated on the last axis, blocked per projection: all q heads, then all k, then all v.
    Tensor qkv;
    std::optional<Tensor> attn_mask;
};

}  // namespace ttnn::prim
