// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.hpp"

#include <variant>

#include "device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/creation.hpp"

#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.hpp"

namespace ttnn::operations::matmul {

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
 * @param transpose_a Whether to transpose the first input tensor
 * @param transpose_b Whether to transpose the second input tensor
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
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt) {
    // Calculate the expected output shape
    ttnn::Shape output_shape =
        utilities::compute_matmul_output_shape(input_tensor_a, input_tensor_b, transpose_a, transpose_b);

    // Use the appropriate data type (either from parameters or from input tensor)
    DataType output_dtype = dtype.value_or(input_tensor_a.dtype());

    // Create a tensor filled with zeros
    auto output_tensor =
        ttnn::full(output_shape, 0.0f, output_dtype, input_tensor_a.layout(), *input_tensor_a.device(), memory_config);

    // Apply bias if provided
    if (bias.has_value()) {
        output_tensor = ttnn::add(output_tensor, bias.value(), std::nullopt, memory_config);
    }

    return output_tensor;
}

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const Activation>& activation) {
    if (!activation.has_value()) {
        return std::nullopt;
    }
    const auto& act = activation.value();
    if (std::holds_alternative<std::string>(act)) {
        return ttnn::operations::unary::utils::string_to_unary_with_param(std::get<std::string>(act));
    }
    return std::get<UnaryWithParam>(act);
}

static bool get_post_process_bias(
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const CoreCoord>& user_core_coord,
    const MemoryConfig& output_mem_config,
    const ttnn::Tensor& input_tensor_a_adjusted,
    const ttnn::Tensor& input_tensor_b_adjusted,
    const bool transpose_a) {
    // Determine if we should post-process bias based on the program config
    // MatmulMultiCoreProgramConfig doesn't support bias fusion, so we need to apply it as a post-process
    bool post_process_bias = false;
    if (bias.has_value()) {
        // Check if bias shape is compatible with kernel fusion
        // Bias fusion requires bias_shape_aligned[-2] == tile_height
        const auto& bias_tensor = bias.value();
        const auto& bias_padded_shape = bias_tensor.padded_shape();
        const auto& tile_shape = input_tensor_a_adjusted.tensor_spec().tile().get_tile_shape();
        uint32_t tile_height = transpose_a ? tile_shape[1] : tile_shape[0];

        // If bias second-to-last dimension doesn't match tile height, must post-process
        if (bias_padded_shape[-2] != tile_height) {
            post_process_bias = true;
        } else if (program_config.has_value()) {
            // Check if the provided program config is MatmulMultiCoreProgramConfig
            post_process_bias = std::holds_alternative<MatmulMultiCoreProgramConfig>(program_config.value());
        } else if (!user_core_coord.has_value()) {
            // When program_config and user_core_coord are not provided, config is auto-generated

            // Special case: L1 memory often leads to MatmulMultiCoreProgramConfig
            // Be conservative and post-process bias for non-DRAM outputs
            if (output_mem_config.buffer_type() != BufferType::DRAM) {
                post_process_bias = true;
            } else if (!input_tensor_a_adjusted.is_sharded()) {
                // For DRAM output, check if all tensors are DRAM interleaved
                bool all_dram_interleaved =
                    input_tensor_a_adjusted.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                    input_tensor_b_adjusted.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                    output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                    input_tensor_a_adjusted.memory_config().buffer_type() == BufferType::DRAM &&
                    input_tensor_b_adjusted.memory_config().buffer_type() == BufferType::DRAM;

                // If not all DRAM interleaved, MatmulMultiCoreProgramConfig is more likely
                if (!all_dram_interleaved) {
                    post_process_bias = true;
                }
            }
        }
    }
    return post_process_bias;
}

static ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    ttnn::prim::MatmulParams& parameters,
    std::optional<ttnn::Tensor>& optional_output_tensor) {
    if (input_tensor_a.logical_shape().rank() == 0 || input_tensor_b.logical_shape().rank() == 0) [[unlikely]] {
        TT_THROW(
            "ttnn.matmul: Both arguments to matmul need to be at least 1D, but got shapes {} and {}",
            input_tensor_a.logical_shape(),
            input_tensor_b.logical_shape());
    }

    if (input_tensor_a.is_sharded() || input_tensor_b.is_sharded()) {
        TT_FATAL(
            !parameters.user_fused_activation.has_value(),
            "Sharded matmul run with {} activation: this should be placed in the program config's fused_activation "
            "field",
            parameters.user_fused_activation.value().op_type);
    }

    // Check for zero volume tensors
    if (input_tensor_a.logical_volume() == 0 || input_tensor_b.logical_volume() == 0) [[unlikely]] {
        return detail::handle_zero_volume_matmul(
            input_tensor_a,
            input_tensor_b,
            parameters.output_mem_config,
            parameters.output_dtype,
            parameters.transpose_a,
            parameters.transpose_b,
            bias);
    }

    //----------------------------------------------------------------------------------------------
    // The following code is replicated from matmul_op.cpp and helps determine the program config
    auto matmul_struct =
        ttnn::prim::create_matmul_attributes(input_tensor_a, input_tensor_b, parameters, {optional_output_tensor});

    uint32_t bias_single_tile_size = 0;
    if (bias.has_value()) {
        auto bias_data_format = datatype_to_dataformat_converter(bias.value().dtype());
        bias_single_tile_size = tt::tile_size(bias_data_format);
    }
    MatmulProgramConfig chosen_program_config = get_program_config(
        input_tensor_a,
        input_tensor_b,
        parameters.transpose_a,
        parameters.transpose_b,
        bias_single_tile_size,
        matmul_struct);
    //----------------------------------------------------------------------------------------------

    // Decide if we need to manually transpose or if the program config will handle it
    bool needs_manual_transpose =
        !(std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config) ||
          std::holds_alternative<MatmulMultiCoreReuseMultiCastProgramConfig>(chosen_program_config) ||
          std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(chosen_program_config));
    bool needs_manual_transpose_a = parameters.transpose_a && needs_manual_transpose;
    bool needs_manual_transpose_b = parameters.transpose_b && needs_manual_transpose;

    const auto& input_tensor_a_adjusted = needs_manual_transpose_a
                                              ? ttnn::transpose(input_tensor_a, -1, -2, input_tensor_a.memory_config())
                                              : input_tensor_a;

    ttnn::Tensor input_tensor_b_adjusted = input_tensor_b;
    if (input_tensor_b.logical_shape().rank() == 1) {
        input_tensor_b_adjusted = ttnn::reshape(input_tensor_b, ttnn::Shape({input_tensor_b.logical_shape()[-1], 1}));
    } else if (needs_manual_transpose_b) {
        input_tensor_b_adjusted = ttnn::transpose(input_tensor_b, -1, -2, input_tensor_b.memory_config());
    }

    // We need to change the transpose_a and transpose_b flags if we manually transposed
    // the input tensors
    if (needs_manual_transpose_a) {
        parameters.transpose_a = false;
    }
    if (needs_manual_transpose_b) {
        parameters.transpose_b = false;
    }

    bool post_process_bias = get_post_process_bias(
        bias,
        parameters.program_config,
        parameters.user_core_coord,
        parameters.output_mem_config,
        input_tensor_a_adjusted,
        input_tensor_b_adjusted,
        parameters.transpose_a);

    auto attributes = ttnn::prim::create_matmul_attributes(
        input_tensor_a_adjusted, input_tensor_b_adjusted, parameters, {optional_output_tensor});

    auto output_tensor = ttnn::prim::matmul(
                             input_tensor_a_adjusted,
                             input_tensor_b_adjusted,
                             post_process_bias ? std::nullopt : bias,
                             optional_output_tensor,
                             attributes)
                             .at(0);

    if (input_tensor_b.logical_shape().rank() == 1) [[unlikely]] {
        output_tensor = ttnn::reshape(
            output_tensor,
            utilities::compute_matmul_output_shape(
                input_tensor_a, input_tensor_b, attributes.transpose_a, attributes.transpose_b));
    }

    // Apply bias as post-processing if needed
    if (post_process_bias) {
        output_tensor = ttnn::add(
            output_tensor,
            bias.value(),
            /*output_dtype=*/std::nullopt,
            output_tensor.memory_config(),
            optional_output_tensor);
    }

    if (parameters.user_fused_activation.has_value() && !parameters.user_core_coord.has_value()) {
        const UnaryWithParam& activation = parameters.user_fused_activation.value();

        output_tensor = ttnn::operations::unary::Unary_chain::invoke(
            output_tensor, {activation}, output_tensor.memory_config(), optional_output_tensor);
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
    const std::optional<const Activation>& activation,
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
    auto matmul_params = ttnn::prim::MatmulParams{
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
        sub_device_id};

    return bound_matmul(
        input_tensor_a,
        input_tensor_b,
        /*bias=*/std::nullopt,
        matmul_params,
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
    const std::optional<const Activation>& activation,
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

    auto matmul_params = ttnn::prim::MatmulParams{
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
        sub_device_id};
    return bound_matmul(input_tensor_a, input_tensor_b, bias, matmul_params, optional_output_tensor);
}

std::vector<Tensor> MatmulBatchedWeightsOperation::invoke(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const Activation>& activation,
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

    std::vector<Tensor> input_tensors = input_tensors_b;
    input_tensors.insert(input_tensors.begin(), input_tensor_a);

    auto parameters = ttnn::prim::MatmulParams{
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
        sub_device_id};

    return ttnn::prim::matmul(
        input_tensors,
        optional_output_tensor,
        ttnn::prim::create_matmul_attributes(input_tensor_a, input_tensors_b[0], parameters, {optional_output_tensor}));
}

void AddmmOperation::validate(
    const Tensor& input_tensor, const Tensor& mat1_tensor, const Tensor& mat2_tensor, float alpha, float beta) {
    TT_FATAL(alpha != 0.0, "alpha parameter cannot be 0");

    if (beta != 0.0) {
        const auto& input_shape = input_tensor.logical_shape();
        const auto& mat1_shape = mat1_tensor.logical_shape();
        const auto& mat2_shape = mat2_tensor.logical_shape();

        TT_FATAL(
            input_shape[0] == mat1_shape[0] && input_shape[1] == mat2_shape[1],
            "input_tensor must have shape matching one of result of mat1_tensor @ mat2_tensor");

        auto idtype = input_tensor.dtype();
        TT_FATAL(
            idtype == DataType::BFLOAT16 || idtype == DataType::FLOAT32 || idtype == DataType::BFLOAT8_B,
            "only ttnn.bfloat16, ttnn.float32 and ttnn.bfloat8_b types are supported for input_tensor");
    }

    auto m1type = mat1_tensor.dtype();
    TT_FATAL(
        m1type == DataType::BFLOAT16 || m1type == DataType::FLOAT32 || m1type == DataType::BFLOAT8_B,
        "only ttnn.bfloat16, ttnn.float32 and ttnn.bfloat8_b types are supported for mat1_tensor");

    auto m2type = mat2_tensor.dtype();
    TT_FATAL(
        m2type == DataType::BFLOAT16 || m2type == DataType::FLOAT32 || m2type == DataType::BFLOAT8_B,
        "only ttnn.bfloat16, ttnn.float32 and ttnn.bfloat8_b types are supported for mat2_tensor");
}

Tensor AddmmOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& mat1_tensor,
    const Tensor& mat2_tensor,
    float alpha,
    float beta,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(!output_tile.has_value(), "output_tile must not be provided");

    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    validate(input_tensor, mat1_tensor, mat2_tensor, alpha, beta);

    auto matmul_params = ttnn::prim::MatmulParams{
        program_config,
        std::nullopt,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        /*untilize_out=*/false,
        /*user_core_coord=*/user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        output_tile,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/std::nullopt};
    auto out_tensor = bound_matmul(mat1_tensor, mat2_tensor, std::nullopt, matmul_params, optional_output_tensor);

    if (alpha != 1.0) {
        multiply_(out_tensor, alpha);
    }

    if (beta != 0.0) {
        auto add_tensor = beta != 1.0 ? multiply(input_tensor, beta) : input_tensor;
        add_(out_tensor, add_tensor);
    }

    return out_tensor;
}

Tensor SparseMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const std::optional<uint32_t> nnz,
    bool is_input_a_sparse,
    bool is_input_b_sparse,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<CoreCoord> user_core_coord =
        core_grid.has_value() ? std::make_optional(CoreCoord(core_grid->x, core_grid->y)) : std::nullopt;

    return ttnn::prim::sparse_matmul(
               input_tensor_a,
               input_tensor_b,
               sparsity,
               optional_output_tensor,
               nnz,
               is_input_a_sparse,
               is_input_b_sparse,
               memory_config,
               dtype,
               program_config,
               compute_kernel_config,
               user_core_coord,
               output_tile,
               global_cb,
               sub_device_id)
        .at(0);
}

}  // namespace ttnn::operations::matmul
