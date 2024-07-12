// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tt_dnn/op_library/groupnorm/groupnorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

struct GroupNorm {

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const int num_groups,
        const float epsilon,
        const std::optional<ttnn::Tensor>& input_mask = std::nullopt,
        const std::optional<ttnn::Tensor>& weight = std::nullopt,
        const std::optional<ttnn::Tensor>& bias = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<CoreGrid> core_grid = std::nullopt,
        std::optional<bool> inplace = std::nullopt,
        std::optional<ttnn::Layout> output_layout = std::nullopt) {
        if (input_tensor.get_layout() == Layout::TILE and inplace.has_value()) {
            TT_FATAL(inplace == false, "Tile layour does not support inplace tensors");
        }
        if (output_layout.has_value() and inplace.has_value()) {
            if (output_layout != input_tensor.get_layout()) {
                TT_FATAL(inplace == false, "cannot inplace tensors when layout are different");
            }
        }
        TT_FATAL(core_grid.has_value(), "Automatic determination of grid size not supported");

        TT_FATAL(input_tensor.is_sharded(), "Only sharded input tensors supported");

        TT_FATAL(
            input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor cannot be width sharded");

        TT_FATAL(input_tensor.get_shape().rank() == 4, "Input tensor must be rank 4");

        TT_FATAL(
            input_tensor.get_shape()[-1] % num_groups == 0, "Number of channels must be divisible by number of groups");

        const auto& ts = input_tensor.get_shape();
        TT_FATAL(
            (ts[0] * ts[1] * ts[2]) % ttnn::types::TILE_SIZE == 0,
            "Input tensor dim NHW must be divisible by tile size");

        const auto output_dtype = dtype.value_or(input_tensor.get_dtype());

        const std::optional<ttnn::Tensor>& gamma =
            weight.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(weight.value())) : std::nullopt;
        const std::optional<ttnn::Tensor>& beta =
            bias.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(bias.value())) : std::nullopt;

        const MemoryConfig& dram_memory_config = tt::tt_metal::MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            .buffer_type = tt::tt_metal::BufferType::DRAM};
        const MemoryConfig& output_mem_config = memory_config.value_or(dram_memory_config);

        const tt::operations::primary::GroupNormShardedMultiCoreProgramConfig& program_config = {
            .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
            .math_fidelity = MathFidelity::HiFi4,
            .im_data_format = DataType::BFLOAT16,
            .out_data_format = DataType::BFLOAT16,
            .inplace = inplace.value_or(false),
            .output_layout = output_layout.value_or(input_tensor.get_layout())};

        return tt::operations::primary::groupnorm(
            input_tensor, num_groups, epsilon, gamma, beta, input_mask, output_mem_config, program_config);
    }
};

}  // namespace normalization
}  // namespace operations

constexpr auto group_norm = ttnn::register_operation<ttnn::operations::normalization::GroupNorm>("ttnn::group_norm");

}  // namespace ttnn
