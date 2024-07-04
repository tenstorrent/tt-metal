// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.hpp"

#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"

namespace ttnn {

using MatmulMultiCoreReuseProgramConfig = tt::operations::primary::MatmulMultiCoreReuseProgramConfig;
using MatmulMultiCoreReuseMultiCastProgramConfig = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig;
using MatmulMultiCoreReuseMultiCast1DProgramConfig =
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig;
// MatmulProgramConfig is the Union of the above types
using MatmulProgramConfig = tt::operations::primary::MatmulProgramConfig;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape) {
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


std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation) {
    if (!activation.has_value()) {
        return std::nullopt;
    }
    return string_to_unary_with_param(activation.value());
}

ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MatmulProgramConfig> program_config,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid,
    const bool propagate_is_b_batched) {
    const auto& input_tensor_a_adjusted = transpose_a ? tt::tt_metal::transpose(input_tensor_a, -1, -2, input_tensor_a.memory_config()) : input_tensor_a;
    const auto& input_tensor_b_adjusted = transpose_b ? tt::tt_metal::transpose(input_tensor_b, -1, -2, input_tensor_b.memory_config()) : input_tensor_b;

    const auto input_tensor_a_shape = input_tensor_a_adjusted.get_shape();
    const auto input_tensor_b_shape = input_tensor_b_adjusted.get_shape();
    const auto a_padded_shape = input_tensor_a_shape.with_tile_padding();
    const auto b_padded_shape = input_tensor_b_shape.with_tile_padding();

    const auto padded_width_a = a_padded_shape[-1];
    const auto padded_height_b = b_padded_shape[-2];

    if (padded_width_a != padded_height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);
    bool batch_with_bias = input_b_is_batched && bias.has_value();
    TT_FATAL(!batch_with_bias, "Batched input not supported when bias exists (linear operation).");

    std::optional<CoreCoord> user_core_coord;
    const bool has_user_grid = core_grid.has_value();
    if (has_user_grid) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    const bool has_program_config = program_config.has_value();
    bool post_process_bias = false;
    if (bias.has_value()) {
        if (!has_program_config && !has_user_grid) {
            post_process_bias = true;
        }
    }

    auto output_tensor = tt::operations::primary::matmul(
        input_tensor_a_adjusted,
        input_tensor_b_adjusted,
        post_process_bias ? std::nullopt : bias,
        program_config,
        memory_config,
        dtype,
        compute_kernel_config,
        false /*untilize_out*/,
        user_core_coord,
        get_fused_activation(activation),
        propagate_is_b_batched && input_b_is_batched);

    if (post_process_bias) {
        output_tensor = tt::operations::primary::bcast(
            output_tensor, bias.value(), tt::tt_metal::BcastOpMath::ADD, tt::tt_metal::BcastOpDim::H, memory_config);
    }

    if (activation.has_value() && !has_user_grid) {
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

    return output_tensor;
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
