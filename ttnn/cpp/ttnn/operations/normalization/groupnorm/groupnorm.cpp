// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm.hpp"
#include "device/groupnorm_op.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteGroupNorm::invoke(
        const ttnn::Tensor& input_tensor,
        const int num_groups,
        const float epsilon,
        const std::optional<ttnn::Tensor>& input_mask,
        const std::optional<ttnn::Tensor>& weight,
        const std::optional<ttnn::Tensor>& bias,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<ttnn::DataType> dtype,
        std::optional<CoreGrid> core_grid,
        std::optional<bool> inplace,
        std::optional<ttnn::Layout> output_layout) {

    if (input_tensor.get_layout() == Layout::TILE and inplace.has_value()) {
        TT_FATAL(!inplace.value(),
            "In-place operation not supported: Tile layout requires non-inplace tensors. (inplace={})",
            inplace.value());
    }

    if (output_layout.has_value() and inplace.has_value()) {
        if (output_layout != input_tensor.get_layout()) {
            TT_FATAL(!inplace.value(),
                "In-place operation not allowed: Input and output tensor layouts differ. (input_layout={}, output_layout={})",
                input_tensor.get_layout(), output_layout.value());
        }
    }

    TT_FATAL(core_grid.has_value(),
        "Automatic grid size determination is not supported. Please specify the grid size explicitly.");

    TT_FATAL(input_tensor.is_sharded(),
        "Only sharded input tensors are supported. The provided tensor is not sharded.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout: Input tensor must be width-sharded, but it is not. (memory_layout={})",
        input_tensor.memory_config().memory_layout);

    TT_FATAL(input_tensor.get_shape().rank() == 4,
        "Invalid tensor shape: Input tensor must have rank 4. (rank={})",
        input_tensor.get_shape().rank());

    TT_FATAL(
        input_tensor.get_shape()[-1] % num_groups == 0,
        "Invalid channel configuration: Number of channels ({}) must be divisible by the number of groups ({}).",
        input_tensor.get_shape()[-1], num_groups);

    const auto& ts = input_tensor.get_shape();
    TT_FATAL(
        (ts[0] * ts[1] * ts[2]) % ttnn::types::TILE_SIZE == 0,
        "Invalid tensor dimensions: The product of NHW dimensions ({}) must be divisible by the tile size ({}).",
        ts[0] * ts[1] * ts[2], ttnn::types::TILE_SIZE);


    const auto output_dtype = dtype.value_or(input_tensor.get_dtype());

    const std::optional<ttnn::Tensor>& gamma =
        weight.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(weight.value())) : std::nullopt;
    const std::optional<ttnn::Tensor>& beta =
        bias.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(bias.value())) : std::nullopt;

    const MemoryConfig& dram_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    const MemoryConfig& output_mem_config = memory_config.value_or(dram_memory_config);

    const ttnn::operations::normalization::GroupNormShardedMultiCoreProgramConfig& program_config = {
        .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
        .math_fidelity = MathFidelity::HiFi4,
        .im_data_format = DataType::BFLOAT16,
        .out_data_format = DataType::BFLOAT16,
        .inplace = inplace.value_or(false),
        .output_layout = output_layout.value_or(input_tensor.get_layout())};

    return operation::run(
        GroupNorm{
            .eps=epsilon,
            .num_groups=static_cast<uint32_t>(num_groups),
            .output_mem_config=output_mem_config,
            .program_config=program_config},
            {input_tensor},
            {gamma, beta, input_mask}).at(0);
}


}  // namespace ttnn::operations::normalization
