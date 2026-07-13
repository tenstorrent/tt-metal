// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

struct SDPAParams {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    bool is_causal = false;
    std::optional<int64_t> chunk_start_idx;        // Chunked legacy: scalar offset, part of program cache key
    std::optional<Tensor> chunk_start_idx_tensor;  // Chunked flexible: device tensor [1] int32, read at runtime
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_mla = false;
    std::optional<uint32_t> head_dim_v;
    std::optional<uint32_t> sliding_window_size;
    // Windowed (block-diagonal) attention: when true, the mask is synthesized on-device from the
    // cu_window_seqlens tensor instead of being read from attn_mask. Implies non-causal.
    bool is_windowed = false;
};

struct SDPAInputs {
    Tensor q;
    Tensor k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    // Mirrors SDPAParams::chunk_start_idx_tensor so ProgramDescriptor buffer bindings can patch cache hits.
    std::optional<Tensor> chunk_start_idx_tensor;
    std::optional<Tensor> attention_sink;
    // Cumulative window sequence lengths [num_windows + 1], int32/uint32, ROW_MAJOR. Present only in
    // windowed mode; the writer builds the block-diagonal mask from it.
    std::optional<Tensor> cu_window_seqlens;
};

}  // namespace ttnn::prim
