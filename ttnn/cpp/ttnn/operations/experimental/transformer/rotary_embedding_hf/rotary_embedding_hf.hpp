// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::transformer {

struct RotaryEmbeddingHf {
    static tt::tt_metal::Tensor invoke(
        const tt::tt_metal::Tensor& input_tensor,
        const tt::tt_metal::Tensor& cos_cache,
        const tt::tt_metal::Tensor& sin_cache,
        const bool is_decode,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::experimental {

constexpr auto rotary_embedding_hf = ttnn::register_operation<
    "ttnn::experimental::rotary_embedding_hf",
    ttnn::operations::experimental::transformer::RotaryEmbeddingHf>();

}  // namespace ttnn::experimental
