// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail {

struct SdpaDecodeParams {
    bool is_causal = false;
    bool paged_attention = false;
    std::vector<uint32_t> cur_pos;
    std::optional<float> scale = std::nullopt;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t k_chunk_size = 0;

    std::optional<bool> share_cache = std::nullopt;

    std::optional<bool> use_mla = std::nullopt;
    std::optional<uint32_t> head_dim_v = std::nullopt;

    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry{};

    std::optional<uint32_t> cache_position_modulo = std::nullopt;

    uint32_t spec_multi_pos_tiles = 0;
    bool multi_pos = false;
};

struct SdpaDecodeInputs {
    Tensor q;
    Tensor k;

    std::optional<Tensor> v;

    std::optional<Tensor> cur_pos_tensor;
    std::optional<Tensor> page_table_tensor;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail
