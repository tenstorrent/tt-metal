// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding.hpp"
#include "device/rotary_embedding_device_operation.hpp"

namespace ttnn::operations::transformer {

namespace {
Tensor rotary_embedding(
    const Tensor &input_tensor,
    const Tensor &cos,
    const Tensor &sin,
    std::optional<uint32_t> token_idx = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, cos, sin}))};
    operation::launch_with_autoformat(
        [token_idx, output_mem_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            auto& cos = input_tensors.at(1);
            auto& sin = input_tensors.at(2);
            TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
            uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
            uint32_t B = input_tensor.get_legacy_shape()[0];
            uint32_t X = input_tensor.get_legacy_shape()[-1];
            TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
            TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == X, "Cos dims must match input dims");
            if (token_idx.has_value()) {
                seq_len = input_tensor.get_legacy_shape()[0];
                TT_FATAL(seq_len == 1);
                TT_FATAL(cos.get_legacy_shape()[-2] >= token_idx, "Cos dims must match input dims");
            } else {
                TT_FATAL(cos.get_legacy_shape()[-2] >= seq_len, "Cos dims must match input dims");
            }

            auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            tt::tt_metal::Shape input_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
            FormatParams input_format_params = {.pad_shape = input_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            tt::tt_metal::Shape cos_pad_shape = AutoFormat::pad_to_tile_shape(cos.get_legacy_shape());
            FormatParams cos_format_params = {.pad_shape = cos_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            tt::tt_metal::Shape sin_pad_shape = AutoFormat::pad_to_tile_shape(sin.get_legacy_shape());
            FormatParams sin_format_params = {.pad_shape = sin_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            return operation::run_with_autoformat(
                    RotaryEmbedding{seq_len, token_idx, output_mem_config, kernel_config_val},
                    {input_tensor, cos, sin},
                    {input_format_params, cos_format_params, sin_format_params},
                    {Layout::TILE});
        }, {input_tensor, cos, sin}, output_tensors);
    return output_tensors.at(0);
}
}

ttnn::Tensor RotaryEmbeddingOperation::operator()(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::optional<uint32_t> token_index,
    const std::optional<MemoryConfig> memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
    uint32_t B = input_tensor.get_legacy_shape()[0];
    uint32_t X = input_tensor.get_legacy_shape()[-1];

    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    return operation::run(
                tt::tt_metal::RotaryEmbedding{
                    seq_len, token_index, memory_config.value_or(input_tensor.memory_config()), kernel_config_val},
                {input_tensor, cos_cache, sin_cache})
        .at(0);
}

}
