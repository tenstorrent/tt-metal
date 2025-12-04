// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk.hpp"

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> RotaryEmbeddingLlamaFusedQKOperation::invoke(
    const Tensor& q_input_tensor,
    const Tensor& k_input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = q_input_tensor.storage_type() == StorageType::DEVICE ? q_input_tensor.device()->arch()
                                                                     : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig q_output_memory_config = q_input_tensor.memory_config();
    tt::tt_metal::MemoryConfig k_output_memory_config = k_input_tensor.memory_config();
    bool row_major_QK = q_input_tensor.layout() == Layout::ROW_MAJOR && k_input_tensor.layout() == Layout::ROW_MAJOR;

    return ttnn::prim::rotary_embedding_llama_fused_qk(
        q_input_tensor,
        k_input_tensor,
        cos_cache,
        sin_cache,
        trans_mat,
        q_output_memory_config,
        k_output_memory_config,
        kernel_config_val,
        row_major_QK);
}

}  // namespace ttnn::operations::experimental::transformer
