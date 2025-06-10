// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk.hpp"

#include "device/rotary_embedding_llama_fused_qk_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> RotaryEmbeddingLlamaFusedQKOperation::invoke(
    const Tensor& q_input_tensor,
    const Tensor& k_input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = q_input_tensor.storage_type() == StorageType::DEVICE
                    ? q_input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig q_output_memory_config = q_input_tensor.memory_config();
    tt::tt_metal::MemoryConfig k_output_memory_config = k_input_tensor.memory_config();
    bool row_major_QK = q_input_tensor.layout() == Layout::ROW_MAJOR && k_input_tensor.layout() == Layout::ROW_MAJOR;
    auto output_tensors = tt::tt_metal::operation::run(
        tt::tt_metal::RotaryEmbeddingLlamaFusedQK{
            q_output_memory_config, k_output_memory_config, kernel_config_val, row_major_QK},
        {q_input_tensor, k_input_tensor, cos_cache, sin_cache, trans_mat});

    return {output_tensors.at(0), output_tensors.at(1)};
}

}  // namespace ttnn::operations::experimental::transformer
