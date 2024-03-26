// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace ttnn {

using MatmulDefaultProgramConfig = tt::operations::primary::MatmulDefaultProgramConfig;
using MatmulMultiCoreReuseProgramConfig = tt::operations::primary::MatmulMultiCoreReuseProgramConfig;
using MatmulMultiCoreReuseMultiCastProgramConfig = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig;
using MatmulMultiCoreReuseMultiCast1DProgramConfig =
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig;
// MatmulProgramConfig is the Union of the above types
using MatmulProgramConfig = tt::operations::primary::MatmulProgramConfig;
namespace operations {
namespace matmul {

namespace detail {

inline bool is_input_batched(const ttnn::Shape& shape) {
    auto is_batched = false;
    for (auto i = 0; i < shape.rank() - 2; ++i) {
        if (shape[i] > 1) {
            is_batched = true;
            break;
        }
    }
    return is_batched;
}

}  // namespace detail

static inline const std::array<ttnn::TensorSchema, 3> input_schemas{
    ttnn::TensorSchema{2, 4, {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, false, false},
    ttnn::TensorSchema{2, 4, {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, false, false},
    ttnn::TensorSchema{2, 4, {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, false, true}};

inline ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const MatmulProgramConfig& program_config,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_a, input_schemas[0]);
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_b, input_schemas[1]);

    const auto input_tensor_a_shape = input_tensor_a.get_shape();
    const auto input_tensor_b_shape = input_tensor_b.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);

    const auto input_tensor_a_4d = ttnn::unsqueeze_to_4D(input_tensor_a);
    const auto input_tensor_b_4d = ttnn::unsqueeze_to_4D(input_tensor_b);

    auto output_tensor = tt::operations::primary::matmul(
        input_tensor_a_4d, input_tensor_b_4d, program_config, memory_config, dtype, compute_kernel_config);

    while (output_tensor.get_shape().rank() != input_tensor_a_shape.rank()) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_tensor_a_shape.rank());
    }
    return output_tensor;
}

inline ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_a, input_schemas[0]);
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_b, input_schemas[1]);

    const auto input_tensor_a_shape = input_tensor_a.get_shape();
    const auto input_tensor_b_shape = input_tensor_b.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);

    const auto input_tensor_a_4d = ttnn::unsqueeze_to_4D(input_tensor_a);
    const auto input_tensor_b_4d = ttnn::unsqueeze_to_4D(input_tensor_b);

    auto compute_interlevead = [&]() {
        if (input_b_is_batched) {
            return tt::tt_metal::bmm(input_tensor_a_4d, input_tensor_b_4d, memory_config, compute_kernel_config);
        } else {
            return tt::tt_metal::matmul(input_tensor_a_4d, input_tensor_b_4d, memory_config, compute_kernel_config);
        }
    };

    auto output_tensor = compute_interlevead();
    while (output_tensor.get_shape().rank() != input_tensor_a_shape.rank()) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_tensor_a_shape.rank());
    }
    return output_tensor;
}

inline ttnn::Tensor linear(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const MatmulProgramConfig& program_config,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    ttnn::validate_input_tensor("ttnn.linear", input_tensor_a, input_schemas[0]);
    ttnn::validate_input_tensor("ttnn.linear", input_tensor_b, input_schemas[1]);
    ttnn::validate_input_tensor("ttnn.linear", bias, input_schemas[2]);

    const auto input_tensor_a_shape = input_tensor_a.get_shape();
    const auto input_tensor_b_shape = input_tensor_b.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);
    TT_ASSERT(input_b_is_batched == false, "Batched input not supported");

    const auto input_tensor_a_4d = ttnn::unsqueeze_to_4D(input_tensor_a);
    const auto input_tensor_b_4d = ttnn::unsqueeze_to_4D(input_tensor_b);

    std::optional<Tensor> bias_4d = std::nullopt;
    if (bias.has_value()) {
        bias_4d = ttnn::unsqueeze_to_4D(bias.value());
    }

    auto output_tensor = tt::operations::primary::matmul(
        input_tensor_a_4d, input_tensor_b_4d, bias_4d, program_config, memory_config, dtype, compute_kernel_config);

    while (output_tensor.get_shape().rank() != input_tensor_a_shape.rank()) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_tensor_a_shape.rank());
    }
    return output_tensor;
}

inline ttnn::Tensor linear(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    ttnn::validate_input_tensor("ttnn.linear", input_tensor_a, input_schemas[0]);
    ttnn::validate_input_tensor("ttnn.linear", input_tensor_b, input_schemas[1]);
    ttnn::validate_input_tensor("ttnn.linear", bias, input_schemas[2]);

    const auto input_tensor_a_shape = input_tensor_a.get_shape();
    const auto input_tensor_b_shape = input_tensor_b.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);
    TT_ASSERT(input_b_is_batched == false, "Batched input not supported");

    const auto input_tensor_a_4d = ttnn::unsqueeze_to_4D(input_tensor_a);
    const auto input_tensor_b_4d = ttnn::unsqueeze_to_4D(input_tensor_b);

    std::optional<Tensor> bias_4d = std::nullopt;
    if (bias.has_value()) {
        bias_4d = ttnn::unsqueeze_to_4D(bias.value());
    }

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto output_tensor = tt::tt_metal::matmul(input_tensor_a_4d, input_tensor_b_4d, memory_config, compute_kernel_config);
    if (bias_4d.has_value()) {
        output_tensor = tt::tt_metal::bcast(
            output_tensor, bias_4d.value(), tt::tt_metal::BcastOpMath::ADD, tt::tt_metal::BcastOpDim::H, memory_config);
    }

    if (activation.has_value()) {
        if (activation.value() == "relu") {
            output_tensor = tt::tt_metal::relu(output_tensor, memory_config);
        } else if (activation.value() == "gelu") {
            output_tensor = tt::tt_metal::gelu(output_tensor, false, memory_config);
        } else if (activation.value() == "silu") {
            output_tensor = tt::tt_metal::silu(output_tensor, memory_config);
        } else {
            TT_THROW("ttnn.matmul: Unsupported activation function");
        }
    }

    while (output_tensor.get_shape().rank() != input_tensor_a_shape.rank()) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_tensor_a_shape.rank());
    }
    return output_tensor;
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
