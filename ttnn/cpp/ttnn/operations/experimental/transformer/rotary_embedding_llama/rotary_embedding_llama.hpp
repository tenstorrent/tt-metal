// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

 struct RotaryEmbeddingLlamaOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const Tensor& trans_mat,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
 };

}  // namespace operations::::experimental::transformer

namespace experimental {

constexpr auto rotary_embedding_llama = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rotary_embedding_llama",
    ttnn::operations::experimental::transformer::RotaryEmbeddingLlamaOperation>();

}  // namespace experimental

}  // namespace ttnn
