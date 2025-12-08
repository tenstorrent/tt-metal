// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama.hpp"

#include "device/rotary_embedding_llama_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

Tensor RotaryEmbeddingLlamaOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    const bool is_decode_mode,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::rotary_embedding_llama(
        input_tensor, cos_cache, sin_cache, trans_mat, is_decode_mode, memory_config, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer
