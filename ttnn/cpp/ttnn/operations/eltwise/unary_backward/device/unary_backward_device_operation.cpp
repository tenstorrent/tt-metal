// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

#include "unary_backward_op_utils.hpp"
#include "unary_backward_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary_backward {

namespace {

// Unary gradients are floating-point math; the integer dtypes have no meaningful derivative.
bool is_supported_dtype(DataType dtype) {
    return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT8_B ||
           dtype == DataType::BFLOAT4_B;
}

// Properties every operand of every op in UnaryBackwardOpType must have, because the shared
// program factory reads interleaved tile pages on device with a work split derived from the
// input's tile count.
void validate_operand(std::string_view op_name, const Tensor& tensor, std::string_view name) {
    TT_FATAL(
        is_supported_dtype(tensor.dtype()),
        "{} operation only supports floating-point dtypes (bfloat16, float32, bfloat8_b, bfloat4_b), but {} has "
        "dtype {}.",
        op_name,
        name,
        tensor.dtype());

    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "{} operation requires {} to be on device, but its storage type is {}.",
        op_name,
        name,
        tensor.storage_type());

    TT_FATAL(
        tensor.buffer() != nullptr,
        "{} operation requires {} to be allocated in a buffer on the device, but the buffer is null.",
        op_name,
        name);

    TT_FATAL(!tensor.is_sharded(), "{} operation does not support a sharded {}.", op_name, name);

    // The factory sizes its circular buffers with tt::tile_size and splits work by
    // physical_volume() / TILE_HW, and neither the layout nor the tile is in
    // compute_program_hash. The layout also reaches the kernels as the aligned page size
    // inside TensorAccessorArgs, a compile-time arg no cache-hit path can refresh, so a
    // ROW_MAJOR operand would be read as tile pages under a cached key.
    TT_FATAL(
        tensor.layout() == Layout::TILE,
        "{} operation requires {} to be in TILE layout, but it has {} layout.",
        op_name,
        name,
        tensor.layout());

    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "{} operation does not currently support tiles other than 32x32, but {} has a {}x{} tile.",
        op_name,
        name,
        tile.get_height(),
        tile.get_width());

    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} operation requires {} to use INTERLEAVED memory layout, but it uses {}.",
        op_name,
        name,
        tensor.memory_config().memory_layout());
}

}  // namespace

void UnaryBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const std::string_view op_name = to_string(args.op_type);
    const auto& input = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
    const auto& preallocated_input_grad = tensor_args.preallocated_input_grad;

    validate_operand(op_name, input, "the input tensor");
    validate_operand(op_name, grad_output, "the grad_output tensor");

    auto output_memory_config = args.output_memory_config;
    auto output_dtype = args.output_dtype == DataType::INVALID ? input.dtype() : args.output_dtype;
    if (preallocated_input_grad.has_value()) {
        output_memory_config = preallocated_input_grad->memory_config();
        output_dtype = preallocated_input_grad->dtype();
    }

    TT_FATAL(
        output_dtype == input.dtype(),
        "{} operation requires the input and output data types to match. Input data type: {}, output data type: {}",
        op_name,
        input.dtype(),
        output_dtype);

    TT_FATAL(
        input.memory_config().memory_layout() == output_memory_config.memory_layout(),
        "{} operation requires the input and output memory layouts to match. Input layout: {}, output layout: {}",
        op_name,
        input.memory_config().memory_layout(),
        output_memory_config.memory_layout());

    // The reader walks the same tile_id range in both operands with a count derived from the
    // input alone (physical_volume() / TILE_HW), so a smaller grad_output would be read past
    // the end of its allocation.
    TT_FATAL(
        grad_output.logical_shape() == input.logical_shape(),
        "{} operation requires grad_output and input to have the same logical shape, but got {} and {}.",
        op_name,
        grad_output.logical_shape(),
        input.logical_shape());
    TT_FATAL(
        grad_output.padded_shape() == input.padded_shape(),
        "{} operation requires grad_output and input to have the same padded shape, but got {} and {}.",
        op_name,
        grad_output.padded_shape(),
        input.padded_shape());

    if (preallocated_input_grad.has_value()) {
        const auto& preallocated = preallocated_input_grad.value();
        validate_operand(op_name, preallocated, "the preallocated output tensor");
        // Pin the preallocated output to the input rather than to compute_output_specs: that
        // function returns this very tensor's spec when one is supplied, so comparing against
        // it is a tautology that can never fire. The writer emits one page per input tile at
        // an offset derived from input.physical_volume(), so an undersized buffer would be
        // written past its end.
        TT_FATAL(
            preallocated.logical_shape() == input.logical_shape(),
            "{} operation requires a preallocated output tensor to have the same logical shape as the input. Input "
            "shape: {}, preallocated output shape: {}",
            op_name,
            input.logical_shape(),
            preallocated.logical_shape());
        TT_FATAL(
            preallocated.padded_shape() == input.padded_shape(),
            "{} operation requires a preallocated output tensor to have the same padded shape as the input, because "
            "the writer emits one page per input tile. Input padded shape: {}, preallocated output padded shape: {}",
            op_name,
            input.padded_shape(),
            preallocated.padded_shape());
    }
}

UnaryBackwardDeviceOperation::spec_return_value_t UnaryBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return tensor_args.preallocated_input_grad->tensor_spec();
    }

    const DataType output_dtype =
        args.output_dtype == DataType::INVALID ? tensor_args.input.dtype() : args.output_dtype;

    return TensorSpec(
        tensor_args.input.logical_shape(), TensorLayout(output_dtype, Layout::TILE, args.output_memory_config));
}

UnaryBackwardDeviceOperation::tensor_return_value_t UnaryBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return *tensor_args.preallocated_input_grad;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t UnaryBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;

    // args carries op_type, so entries for two different gradients can never collide.
    operation::Hash hash = operation::hash_operation<UnaryBackwardDeviceOperation>(
        args,
        input.dtype(),
        input.memory_config(),
        grad_output.dtype(),
        grad_output.memory_config(),
        input.padded_shape().volume());

    // args only carries the requested output_dtype/output_memory_config; when the caller
    // supplies its own output tensor that is what the factory actually binds, sizing the
    // destination CB from its dtype and baking a TensorAccessorArgs for its buffer into the
    // writer's compile-time args. Neither can be refreshed on a cache hit, so key on the
    // tensor that is really used.
    if (tensor_args.preallocated_input_grad.has_value()) {
        const auto& preallocated = tensor_args.preallocated_input_grad.value();
        hash =
            ttsl::hash::hash_objects(hash, preallocated.dtype(), preallocated.layout(), preallocated.memory_config());
    }

    return hash;
}

Tensor launch_unary_backward(
    UnaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_input_grad) {
    auto operation_attributes = UnaryBackwardDeviceOperation::operation_attributes_t{
        .op_type = op_type, .output_dtype = output_dtype, .output_memory_config = output_memory_config};
    auto tensor_args = UnaryBackwardDeviceOperation::tensor_args_t{
        .grad_output = grad_output, .input = input, .preallocated_input_grad = preallocated_input_grad};

    return ttnn::device_operation::launch<UnaryBackwardDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::unary_backward
