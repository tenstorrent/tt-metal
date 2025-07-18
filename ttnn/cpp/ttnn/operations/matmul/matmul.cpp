// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn {

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape) {
    if (shape.rank() < 2) [[unlikely]] {
        return false;
    }

    auto is_batched = false;
    for (auto i = 0; i < shape.rank() - 2; ++i) {
        if (shape[i] > 1) {
            is_batched = true;
            break;
        }
    }
    return is_batched;
}

/**
 * @brief Handles matmul operations with zero volume inputs by creating a zero-filled output tensor
 *
 * When one of the input tensors has zero volume (a dimension with size 0), this function:
 * 1. Computes the correct output shape using compute_matmul_output_shape
 * 2. Creates an output tensor with that shape, filled with zeros
 * 3. Optionally adds bias to the output tensor
 *
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @param memory_config Memory configuration for the output tensor
 * @param dtype Data type for the output tensor
 * @param bias Optional bias tensor to add to the result
 * @return Zero-filled tensor with the appropriate output shape, with bias applied if provided
 */
Tensor handle_zero_volume_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MemoryConfig& memory_config,
    const std::optional<DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt) {
    // Calculate the expected output shape
    ttnn::Shape output_shape = compute_matmul_output_shape(input_tensor_a, input_tensor_b);

    // Use the appropriate data type (either from parameters or from input tensor)
    DataType output_dtype = dtype.value_or(input_tensor_a.dtype());

    // Create a tensor filled with zeros
    auto output_tensor = ttnn::full(
        output_shape, 0.0f, output_dtype, input_tensor_a.layout(), *input_tensor_a.mesh_device(), memory_config);

    // Apply bias if provided
    if (bias.has_value()) {
        output_tensor = ttnn::add(output_tensor, bias.value(), std::nullopt, memory_config);
    }

    return output_tensor;
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
    const uint8_t& queue_id,
    std::optional<ttnn::Tensor>& optional_output_tensor) {
    if (input_tensor_a.logical_shape().rank() == 0 || input_tensor_b.logical_shape().rank() == 0) [[unlikely]] {
        TT_THROW(
            "ttnn.matmul: Both arguments to matmul need to be at least 1D, but got shapes {} and {}",
            input_tensor_a.logical_shape(),
            input_tensor_b.logical_shape());
    }

    // Check for zero volume tensors
    if (input_tensor_a.logical_volume() == 0 || input_tensor_b.logical_volume() == 0) [[unlikely]] {
        return detail::handle_zero_volume_matmul(
            input_tensor_a, input_tensor_b, parameters.output_mem_config, parameters.output_dtype, bias);
    }

    const auto& input_tensor_a_adjusted = parameters.transpose_a
                                              ? ttnn::transpose(input_tensor_a, -1, -2, input_tensor_a.memory_config())
                                              : input_tensor_a;
    const auto& input_tensor_b_adjusted =
        (input_tensor_b.logical_shape().rank() == 1)
            ? ttnn::reshape(input_tensor_b, ttnn::Shape({input_tensor_b.logical_shape()[-1], 1}))
        : parameters.transpose_b ? ttnn::transpose(input_tensor_b, -1, -2, input_tensor_b.memory_config())
                                 : input_tensor_b;

    const auto input_tensor_a_shape = input_tensor_a_adjusted.logical_shape();
    const auto input_tensor_b_shape = input_tensor_b_adjusted.logical_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW(
            "ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor ({} != {}). "
            "The shape of first tensor was {} and the shape of second tensor was {})",
            width_a,
            height_b,
            input_tensor_a_shape,
            input_tensor_b_shape);
    }

    const bool has_program_config = parameters.program_config.has_value();
    const bool has_user_grid = parameters.user_core_coord.has_value();
    bool post_process_bias = false;
    if (bias.has_value()) {
        if (!has_program_config && !has_user_grid) {
            post_process_bias = true;
        }
    }

    auto output_tensor = matmul(
        input_tensor_a_adjusted,
        input_tensor_b_adjusted,
        post_process_bias ? std::nullopt : bias,
        parameters,
        DefaultQueueId,
        optional_output_tensor);

    if (input_tensor_b.logical_shape().rank() == 1) [[unlikely]] {
        output_tensor = ttnn::reshape(
            output_tensor, ttnn::operations::matmul::compute_matmul_output_shape(input_tensor_a, input_tensor_b));
    }

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
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool user_run_batched = detail::is_input_batched(input_tensor_b.logical_shape());
    const bool untilize_out =
        program_config.has_value() &&
                std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config.value())
            ? std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config.value()).untilize_out
            : false;
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
            untilize_out,
            user_core_coord,
            get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            output_tile,
            global_cb,
            sub_device_id},
        /*queue_id=*/0,
        optional_output_tensor);
}

Tensor LinearOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    std::optional<ttnn::Tensor> optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool b_is_batched = detail::is_input_batched(input_tensor_b.logical_shape());
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
            transpose_b,
            output_tile,
            global_cb,
            sub_device_id},
        /*queue_id=*/0,
        optional_output_tensor);
}

std::vector<Tensor> MatmulBatchedWeightsOperation::invoke(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    TT_FATAL(transpose_a == false, "cannot transpose A in batched matmul");
    TT_FATAL(transpose_b == false, "cannot transpose B in batched matmul");
    TT_FATAL(memory_config.has_value(), "memory_config must be provided");
    TT_FATAL(program_config.has_value(), "program_config must be provided");
    TT_FATAL(!activation.has_value(), "activation must not be provided");
    TT_FATAL(!core_grid.has_value(), "core_grid must not be provided");
    TT_FATAL(!output_tile.has_value(), "output_tile must not be provided");
    TT_FATAL(!optional_output_tensor.has_value(), "optional_output_tensor must not be provided");
    TT_FATAL(global_cb.has_value(), "global_cb must be provided");
    TT_FATAL(sub_device_id.has_value(), "sub_device_id must be provided");

    return matmul_batched_weights(
        input_tensor_a,
        input_tensors_b,
        /*bias=*/std::nullopt,
        Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
            dtype,
            compute_kernel_config,
            /*untilize_out=*/false,
            /*user_core_coord*/ std::nullopt,
            get_fused_activation(activation),
            /*user_run_batched=*/false,
            transpose_a,
            transpose_b,
            output_tile,
            global_cb,
            sub_device_id},
        DefaultQueueId,
        optional_output_tensor);
}

Tensor SparseMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    uint32_t nnz,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<CoreCoord> user_core_coord =
        core_grid.has_value() ? std::make_optional(CoreCoord(core_grid->x, core_grid->y)) : std::nullopt;
    return sparse_matmul(
        input_tensor_a,
        input_tensor_b,
        sparsity,
        SparseMatmul{
            nnz,
            program_config,
            memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
            dtype,
            compute_kernel_config,
            user_core_coord,
            output_tile,
            global_cb,
            sub_device_id},
        DefaultQueueId,
        optional_output_tensor);
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
