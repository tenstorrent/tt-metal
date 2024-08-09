// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::transformer {

 struct RotaryEmbeddingLlamaOperation {
    static ttnn::Tensor operator()(
        const Tensor& input_tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const Tensor trans_mat,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
 };

}  // namespace operations::transformer

namespace transformer {

constexpr auto rotary_embedding_llama = ttnn::register_operation_with_auto_launch_op<
    "ttnn::transformer::rotary_embedding_llama",
    ttnn::operations::transformer::RotaryEmbeddingLlamaOperation>();

}  // namespace transformer

}  // namespace ttnn
