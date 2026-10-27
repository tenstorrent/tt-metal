// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

struct PostCombineReduceParams {
    uint32_t expert_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
    // L1 address the circular buffers must end below (0: the whole L1); with a residual that would not fit, the
    // residual input is single-buffered. Sizes the CBs, so part of the program-cache key.
    uint32_t l1_end = 0;
};

struct PostCombineReduceInputs {
    ttnn::Tensor combine_output;
    ttnn::Tensor weights;
    // Optional tensors driving the dispatch-table-based expert skip (DeepSeek).
    // When omitted, the kernel falls back to skipping experts whose routing
    // weight is exactly zero (GPT-OSS). Both must be supplied together.
    std::optional<ttnn::Tensor> indices;
    std::optional<ttnn::Tensor> expert_dispatch_table;
    // Optional [.., seq_len, emb_dim] bf16 TILE tensor added to the reduced output (e.g. a shared-expert partial).
    std::optional<ttnn::Tensor> residual;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
