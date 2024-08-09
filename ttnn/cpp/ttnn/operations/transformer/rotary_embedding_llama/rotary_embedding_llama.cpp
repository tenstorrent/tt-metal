// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama.hpp"

#include "device/rotary_embedding_llama_device_operation.hpp"

namespace ttnn::operations::transformer {

Tensor RotaryEmbeddingLlamaOperation::operator()(
    const Tensor &input_tensor,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor& trans_mat,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, cos, sin, trans_mat}))};
    operation::launch_op(
        [output_mem_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            uint32_t seq_len = input_tensor.get_legacy_shape()[-2];

            auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            return operation::run(
                    RotaryEmbeddingLlama{seq_len, output_mem_config, kernel_config_val}, input_tensors);
        }, {input_tensor, cos, sin, trans_mat}, output_tensors);
    return output_tensors.at(0);
}

}
