// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

struct PostCombineReduceParams {
    uint32_t expert_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
    // Row-indexed combine only: pairs whose row is skip_row are not read (their slot is skipped by the expert-skip
    // test; UINT32_MAX: read every pair).
    uint32_t skip_row = std::numeric_limits<uint32_t>::max();
};

struct PostCombineReduceInputs {
    ttnn::Tensor combine_output;
    ttnn::Tensor weights;
    // Optional tensors driving the dispatch-table-based expert skip (DeepSeek).
    // When omitted, the kernel falls back to skipping experts whose routing
    // weight is exactly zero (GPT-OSS). Both must be supplied together.
    std::optional<ttnn::Tensor> indices;
    std::optional<ttnn::Tensor> expert_dispatch_table;
    // Row-indexed combine: combine_output is a [rows, emb_dim] ROW_MAJOR table and pair (token t, slot j) reads its
    // row combine_row_index[t, j] (UINT32 ROW_MAJOR [..., seq_len, num_experts_per_tok], one page per token), instead
    // of a gathered [..., seq_len, num_experts_per_tok, emb_dim] copy.
    std::optional<ttnn::Tensor> combine_row_index;
};

// The [..., seq_len, num_experts_per_tok, emb_dim] shape the reduce works on: combine_output's, or with
// combine_row_index the index shape followed by combine_output's emb_dim.
inline ttnn::Shape logical_combine_shape(const PostCombineReduceInputs& t) {
    if (!t.combine_row_index.has_value()) {
        return t.combine_output.padded_shape();
    }
    const auto& idx = t.combine_row_index->logical_shape();
    ttsl::SmallVector<uint32_t> dims(idx.cbegin(), idx.cend());
    dims.push_back(t.combine_output.padded_shape()[-1]);
    return ttnn::Shape(dims);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
