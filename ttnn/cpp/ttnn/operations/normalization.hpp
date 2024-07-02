// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "ttnn/experimental//tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "ttnn/experimental//tt_dnn/op_library/layernorm/layernorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

template <bool in_place>
struct Softmax {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const int dim_arg,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();
        auto rank = input_shape.size();
        auto dim = dim_arg;
        if (dim < 0) {
            dim = rank + dim;
        }

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        auto is_tile_padded = input_tensor.get_shape()[-2] != input_tensor.get_shape().with_tile_padding()[-2] or
                              input_tensor.get_shape()[-1] != input_tensor.get_shape().with_tile_padding()[-1];
        if (dim == rank - 1) {
            auto output_tensor =
                tt::tt_metal::softmax(input_tensor_4D, memory_config.value_or(input_tensor.memory_config()));
            return ttnn::reshape(output_tensor, input_shape);
        } else {
            auto dim_4D = dim + 4 - rank;
            auto output_tensor = tt::operations::primary::moreh_softmax(input_tensor_4D, dim_4D);
            return ttnn::reshape(output_tensor, input_shape);
        }
    }
};

struct LayerNorm {
    static inline const std::array<ttnn::TensorSchema, 4> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                1,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
                true,
                false,
                false,
                true},
            ttnn::TensorSchema{
                1,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
                true,
                false,
                false,
                true},
            ttnn::TensorSchema{
                1,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
                true,
                false,
                false,
                true}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(
        const Tensor& input_tensor,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        Args&&... args) {
        return std::forward_as_tuple(input_tensor, weight, bias, residual_input_tensor);
    }

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<const LayerNormProgramConfig>& program_config_arg = std::nullopt) {
        const LayerNormProgramConfig& program_config = program_config_arg.value_or(LayerNormDefaultProgramConfig{});

        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        if (residual_input_tensor.has_value()) {
            return tt::operations::primary::add_layernorm(
                input_tensor, residual_input_tensor.value(), epsilon, weight, bias, memory_config, program_config);
        } else {
            return tt::operations::primary::layernorm(
                input_tensor, epsilon, weight, bias, memory_config, program_config);
        }
    }
};

struct RMSNorm {
    static inline const std::array<ttnn::TensorSchema, 2> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                1,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
                true,
                false,
                false,
                false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, const Tensor& weight, Args&&... args) {
        return std::forward_as_tuple(input_tensor, weight);
    }

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight,
        float epsilon = 1e-12,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        return tt::operations::primary::rmsnorm(input_tensor, epsilon, weight, std::nullopt, memory_config);
    }
};

struct GroupNorm {
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static inline const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16}, {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT}, true, false, false, false}};
    }

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

constexpr auto softmax = ttnn::register_operation<ttnn::operations::normalization::Softmax<false>>("ttnn::softmax");
constexpr auto layer_norm = ttnn::register_operation<ttnn::operations::normalization::LayerNorm>("ttnn::layer_norm");
constexpr auto rms_norm = ttnn::register_operation<ttnn::operations::normalization::RMSNorm>("ttnn::rms_norm");
constexpr auto group_norm = ttnn::register_operation<ttnn::operations::normalization::GroupNorm>("ttnn::group_norm");
}  // namespace ttnn
