// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "device/rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct RotaryEmbeddingLlamaOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const Tensor& trans_mat,
        bool is_decode_mode = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        ttnn::experimental::prim::RotaryEmbeddingTranspose input_transpose =
            ttnn::experimental::prim::RotaryEmbeddingTranspose::NONE);
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto rotary_embedding_llama = ttnn::register_operation<
    "ttnn::experimental::rotary_embedding_llama",
    ttnn::operations::experimental::transformer::RotaryEmbeddingLlamaOperation>();

}  // namespace experimental

}  // namespace ttnn
