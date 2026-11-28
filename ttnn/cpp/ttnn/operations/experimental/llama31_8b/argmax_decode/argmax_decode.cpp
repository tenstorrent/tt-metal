// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/argmax_decode/argmax_decode.hpp"

#include "ttnn/operations/experimental/llama31_8b/argmax_decode/device/argmax_decode_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor argmax_decode(
    const ttnn::Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor) {
    return ttnn::prim::llama31_8b_argmax_decode(input, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG), output_tensor);
}

}  // namespace ttnn::experimental::llama31_8b
