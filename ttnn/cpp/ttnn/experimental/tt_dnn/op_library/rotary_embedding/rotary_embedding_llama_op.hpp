// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/experimental/tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_llama_multi_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, const Tensor &trans_mat, Tensor &output, DeviceComputeKernelConfig compute_kernel_config);

struct RotaryEmbeddingLlama {
    const uint32_t seq_len;
    const MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

inline Tensor rotary_embedding_llama(
    const Tensor &input_tensor,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor trans_mat,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
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

}  // namespace tt_metal

}  // namespace tt
