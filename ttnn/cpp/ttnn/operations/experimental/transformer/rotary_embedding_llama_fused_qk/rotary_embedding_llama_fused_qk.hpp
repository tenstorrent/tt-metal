// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct RotaryEmbeddingLlamaFusedQKOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& q_input_tensor,
        const Tensor& k_input_tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const Tensor& trans_mat,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto rotary_embedding_llama_fused_qk = ttnn::register_operation<
    "ttnn::experimental::rotary_embedding_llama_fused_qk",
    ttnn::operations::experimental::transformer::RotaryEmbeddingLlamaFusedQKOperation>();

}  // namespace experimental

}  // namespace ttnn
