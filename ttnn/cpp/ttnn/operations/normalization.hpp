// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"

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
        return std::make_tuple(input_tensor);
    }


    template <typename... Args>
    static auto map_launch_op_args_to_execute(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        Args&&... args) {
            return std::make_tuple(input_tensors.at(0), std::forward<Args>(args)...);
    }

    static ttnn::Tensor execute(
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

struct LayerNorm : tt::operations::primary::LayerNorm {
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
        return std::make_tuple(input_tensor, weight, bias, residual_input_tensor);
    }


    template <typename... Args>
    static auto map_launch_op_args_to_execute(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        float epsilon = 1e-12,
        Args&&... args) {
            return std::make_tuple(input_tensors.at(0),epsilon,optional_input_tensors.at(0), optional_input_tensors.at(1), optional_input_tensors.at(2), std::forward<Args>(args)...);
    }

    static inline ttnn::Tensor execute(
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

struct RMSNorm : tt::operations::primary::LayerNorm {
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
        return std::make_tuple(input_tensor, weight);
    }


    template <typename... Args>
    static auto map_launch_op_args_to_execute(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        Args&&... args) {
            return std::make_tuple(input_tensors.at(0), input_tensors.at(1), std::forward<Args>(args)...);
    }

    static inline ttnn::Tensor execute(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight,
        float epsilon = 1e-12,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        return tt::operations::primary::rmsnorm(input_tensor, epsilon, weight, std::nullopt, memory_config);
    }
};

} // namespace normalization
} // namespace operations

constexpr auto softmax = ttnn::register_operation<ttnn::operations::normalization::Softmax<false>>("ttnn::softmax");
constexpr auto layer_norm = ttnn::register_operation<ttnn::operations::normalization::LayerNorm>("ttnn::layer_norm");
constexpr auto rms_norm = ttnn::register_operation<ttnn::operations::normalization::RMSNorm>("ttnn::rms_norm");
} // namespace ttnn
