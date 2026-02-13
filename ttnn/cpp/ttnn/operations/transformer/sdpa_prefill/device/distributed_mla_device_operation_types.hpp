// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

struct DistributedMlaSDPAParams {
    uint32_t device_order = 0;                     // Which device in mesh (0, 1, 2...)
    std::optional<uint32_t> cluster_axis;          // Axis along which to distribute (0 or 1)
    std::optional<float> scale;                    // Scale factor for attention
    tt::tt_metal::MemoryConfig output_mem_config;  // Output memory configuration
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;  // SDPA program config
    DeviceComputeKernelConfig compute_kernel_config;                                 // Compute kernel configuration
    std::optional<int64_t> chunk_start_idx;       // Chunk start index for prefix caching
    bool is_causal = true;                        // Enable causal masking
    bool use_mla = false;                         // Use MLA mode
    std::optional<uint32_t> head_dim_v;           // Value head dimension
    std::optional<uint32_t> sliding_window_size;  // Sliding window size
};

struct DistributedMlaSDPAInputs {
    ttnn::Tensor q;                                      // Full Q tensor (not chunked)
    ttnn::Tensor k;                                      // Full K tensor
    ttnn::Tensor v;                                      // Full V tensor
    std::optional<ttnn::Tensor> attn_mask;               // Attention mask
    std::optional<ttnn::Tensor> page_table;              // Page table for paged KV cache
    std::optional<ttnn::Tensor> attention_sink;          // Attention sink tokens
    std::optional<ttnn::Tensor> chunk_start_idx_tensor;  // Flexible chunk start indices
};

}  // namespace ttnn::operations::transformer::sdpa_prefill
