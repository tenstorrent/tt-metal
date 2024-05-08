// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

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
    };

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
    };

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

constexpr auto layer_norm = ttnn::register_operation<ttnn::operations::normalization::LayerNorm>("ttnn::layer_norm");
constexpr auto rms_norm = ttnn::register_operation<ttnn::operations::normalization::RMSNorm>("ttnn::rms_norm");
} // namespace ttnn
