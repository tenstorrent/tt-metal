// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn {

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
    return ttnn::operations::unary::utils::string_to_unary_with_param(activation.value());
}

ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct Matmul& parameters,
    const uint8_t& queue_id) {
    const auto& input_tensor_a_adjusted = parameters.transpose_a
                                              ? ttnn::transpose(input_tensor_a, -1, -2, input_tensor_a.memory_config())
                                              : input_tensor_a;
    const auto& input_tensor_b_adjusted = parameters.transpose_b
                                              ? ttnn::transpose(input_tensor_b, -1, -2, input_tensor_b.memory_config())
                                              : input_tensor_b;

    const auto input_tensor_a_shape = input_tensor_a_adjusted.get_shape();
    const auto input_tensor_b_shape = input_tensor_b_adjusted.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    const bool has_program_config = parameters.program_config.has_value();
    const bool has_user_grid = parameters.user_core_coord.has_value();
    bool post_process_bias = false;
    if (bias.has_value()) {
        if (!has_program_config && !has_user_grid) {
            post_process_bias = true;
        }
    }

    auto output_tensor =
        matmul(input_tensor_a_adjusted, input_tensor_b_adjusted, post_process_bias ? std::nullopt : bias, parameters);

    if (post_process_bias) {
        output_tensor = ttnn::add(output_tensor, bias.value(), std::nullopt, parameters.output_mem_config);
    }

    if (parameters.user_fused_activation.has_value() && !has_user_grid) {
        const UnaryOpType& op_type = parameters.user_fused_activation.value().op_type;
        if (op_type == UnaryOpType::RELU) {
            output_tensor = ttnn::relu(output_tensor, parameters.output_mem_config);
        } else if (op_type == UnaryOpType::GELU) {
            output_tensor = ttnn::gelu(output_tensor, false, parameters.output_mem_config);
        } else if (op_type == UnaryOpType::SILU) {
            output_tensor = ttnn::silu(output_tensor, parameters.output_mem_config);
        } else {
            TT_THROW("ttnn.matmul: Unsupported activation function");
        }
    }

    return output_tensor;
}

Tensor MatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig> memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig> program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool user_run_batched = detail::is_input_batched(input_tensor_b.get_shape());
    return bound_matmul(
        input_tensor_a,
        input_tensor_b,
        /*bias=*/std::nullopt,
        Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
            dtype,
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b},
        /*queue_id=*/0);
}

Tensor LinearOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig> memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig> program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool b_is_batched = detail::is_input_batched(input_tensor_b.get_shape());
    TT_FATAL(!(b_is_batched && bias.has_value()), "Batched input not supported when bias exists (linear operation).");

    return bound_matmul(
        input_tensor_a,
        input_tensor_b,
        bias,
        Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
            dtype,
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            get_fused_activation(activation),
            /*user_run_batched=*/false,
            transpose_a,
            transpose_b},
        /*queue_id=*/0);
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
