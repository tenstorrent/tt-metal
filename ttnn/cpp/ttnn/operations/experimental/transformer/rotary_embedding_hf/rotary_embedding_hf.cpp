// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf.hpp"
#include "device/rotary_embedding_hf_device_operation.hpp"

namespace ttnn::prim {

tt::tt_metal::Tensor rotary_embedding_hf(
    const tt::tt_metal::Tensor& input,
    const tt::tt_metal::Tensor& cos,
    const tt::tt_metal::Tensor& sin,
    bool is_decode_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim

namespace ttnn::operations::experimental::transformer {

tt::tt_metal::Tensor RotaryEmbeddingHf::invoke(
    const tt::tt_metal::Tensor& input_tensor,
    const tt::tt_metal::Tensor& cos_cache,
    const tt::tt_metal::Tensor& sin_cache,
    const bool is_decode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    auto kernel_config = compute_kernel_config.value_or(ttnn::DeviceComputeKernelConfig::Default());

    return ttnn::prim::rotary_embedding_hf(
        input_tensor, cos_cache, sin_cache, is_decode, output_mem_config, kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer
