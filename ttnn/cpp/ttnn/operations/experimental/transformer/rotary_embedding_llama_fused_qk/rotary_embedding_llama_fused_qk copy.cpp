// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk.hpp"

#include "device/rotary_embedding_llama_fused_qk_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> RotaryEmbeddingLlamaFusedQKOperation::invoke(
    const Tensor &q_input_tensor,
    const Tensor &k_input_tensor,
    const Tensor &cos_cache,
    const Tensor &sin_cache,
    const Tensor& trans_mat,
    const std::optional<MemoryConfig>& q_memory_config,
    const std::optional<MemoryConfig>& k_memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({q_input_tensor, k_input_tensor, cos_cache, sin_cache, trans_mat}))};
    operation::launch_op(
        [q_memory_config, k_memory_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& q_input_tensor = input_tensors.at(0);
            auto& k_input_tensor = input_tensors.at(1);

            auto arch = q_input_tensor.storage_type() == StorageType::DEVICE ? q_input_tensor.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            tt::tt_metal::MemoryConfig default_q_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
            tt::tt_metal::MemoryConfig default_k_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
            if(q_input_tensor.storage_type() == StorageType::DEVICE) {
                default_q_memory_config = q_input_tensor.memory_config();
            }
            if(k_input_tensor.storage_type() == StorageType::DEVICE) {
                default_k_memory_config = k_input_tensor.memory_config();
            }

            return operation::run(
                    RotaryEmbeddingLlamaFusedQK{q_memory_config.value_or(default_q_memory_config), k_memory_config.value_or(default_k_memory_config), kernel_config_val}, input_tensors);
        }, {q_input_tensor, k_input_tensor, cos_cache, sin_cache, trans_mat}, output_tensors);
    return {output_tensors.at(0), output_tensors.at(1)};
}

} // namespace ttnn::operations::experimental::transformer
