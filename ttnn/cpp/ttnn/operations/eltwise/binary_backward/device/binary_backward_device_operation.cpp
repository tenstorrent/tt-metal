// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "binary_backward_op_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_backward {

namespace {

// Common per-operand invariants factored out so grad_output / input / other and any
// preallocated tensor all get the same checks in the same order.
void validate_operand(std::string_view op_name, const Tensor& tensor, std::string_view role) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "{} operation requires {} to be on Device, got storage_type={}",
        op_name,
        role,
        tensor.storage_type());
    TT_FATAL(
        tensor.buffer() != nullptr,
        "{} operation requires {} to be allocated in a buffer on the device; buffer is null",
        op_name,
        role);
    TT_FATAL(
        !tensor.is_sharded(),
        "{} operation does not support sharded {} (memory_layout={})",
        op_name,
        role,
        tensor.memory_config().memory_layout());
    TT_FATAL(
        tensor.layout() == Layout::TILE,
        "{} operation requires {} to be in TILE layout, got {}",
        op_name,
        role,
        tensor.layout());
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "{} operation only supports 32x32 tiles, but {} has a {}x{} tile",
        op_name,
        role,
        tile.get_height(),
        tile.get_width());
}

std::string_view op_name_of(BinaryBackwardOpType op_type) {
    switch (op_type) {
        case BinaryBackwardOpType::MUL_BW: return "MUL_BW";
    }
    return "BINARY_BW";
}

}  // namespace

void BinaryBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto op_name = op_name_of(args.op_type);
    const auto& grad_output = tensor_args.grad_output;
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;

    // MUL_BW's first-cut contract: both grads must be requested. Partial-mask
    // callers fall back to the composite path in binary_backward.cpp.
    TT_FATAL(
        args.are_required_outputs[0] && args.are_required_outputs[1],
        "{} operation currently requires both grads; got are_required_outputs=[{}, {}]",
        op_name,
        args.are_required_outputs[0],
        args.are_required_outputs[1]);

    validate_operand(op_name, grad_output, "the grad_output tensor");
    validate_operand(op_name, input, "the input tensor");
    validate_operand(op_name, other, "the other tensor");

    // Reader walks the same tile_id range in all three operands using a count
    // derived from input alone (physical_volume / TILE_HW); size mismatches would
    // read past the end of grad_output or other.
    TT_FATAL(
        grad_output.padded_shape() == input.padded_shape(),
        "{} operation requires grad_output and input to have the same padded shape, got {} and {}",
        op_name,
        grad_output.padded_shape(),
        input.padded_shape());
    TT_FATAL(
        other.padded_shape() == input.padded_shape(),
        "{} operation requires other and input to have the same padded shape, got {} and {}",
        op_name,
        other.padded_shape(),
        input.padded_shape());

    // Pin all operands to one device before any writer binds their buffer addresses.
    TT_FATAL(
        grad_output.device() == input.device() && other.device() == input.device(),
        "{} operation requires all operands on the same device",
        op_name);

    TT_FATAL(
        input.memory_config().memory_layout() == args.output_memory_config.memory_layout(),
        "{} operation requires input and output memory layouts to match, got input={} vs output={}",
        op_name,
        input.memory_config().memory_layout(),
        args.output_memory_config.memory_layout());
    TT_FATAL(
        args.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} operation requires INTERLEAVED memory layout, got {}",
        op_name,
        args.output_memory_config.memory_layout());

    const auto validate_preallocated =
        [&](const std::optional<Tensor>& preallocated, const Tensor& reference, std::string_view role) {
            if (!preallocated.has_value()) {
                return;
            }
            validate_operand(op_name, preallocated.value(), role);
            TT_FATAL(
                preallocated->device() == input.device(),
                "{} operation requires {} to be on the same device as the input",
                op_name,
                role);
            // Writer emits one page per input tile at an offset derived from
            // input.physical_volume(); an undersized buffer is written past its end.
            TT_FATAL(
                preallocated->logical_shape() == reference.logical_shape(),
                "{} operation requires {} logical shape to match its operand, got {} vs {}",
                op_name,
                role,
                preallocated->logical_shape(),
                reference.logical_shape());
            TT_FATAL(
                preallocated->padded_shape() == reference.padded_shape(),
                "{} operation requires {} padded shape to match its operand, got {} vs {}",
                op_name,
                role,
                preallocated->padded_shape(),
                reference.padded_shape());
        };
    validate_preallocated(tensor_args.preallocated_input_grad, input, "the preallocated input_grad tensor");
    validate_preallocated(tensor_args.preallocated_other_grad, other, "the preallocated other_grad tensor");
}

BinaryBackwardDeviceOperation::spec_return_value_t BinaryBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto spec_for = [&](const Tensor& ref, const std::optional<Tensor>& preallocated) -> TensorSpec {
        if (preallocated.has_value()) {
            return preallocated->tensor_spec();
        }
        DataType output_dtype = args.output_dtype;
        if (output_dtype == DataType::INVALID) {
            output_dtype = ref.dtype();
        }
        return TensorSpec(ref.logical_shape(), TensorLayout(output_dtype, Layout::TILE, args.output_memory_config));
    };
    return {
        spec_for(tensor_args.input, tensor_args.preallocated_input_grad),
        spec_for(tensor_args.other, tensor_args.preallocated_other_grad),
    };
}

BinaryBackwardDeviceOperation::tensor_return_value_t BinaryBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input.device();
    tensor_return_value_t outputs;
    outputs.reserve(2);
    outputs.push_back(
        tensor_args.preallocated_input_grad.has_value() ? *tensor_args.preallocated_input_grad
                                                        : create_device_tensor(specs[0], device));
    outputs.push_back(
        tensor_args.preallocated_other_grad.has_value() ? *tensor_args.preallocated_other_grad
                                                        : create_device_tensor(specs[1], device));
    return outputs;
}

ttsl::hash::hash_t BinaryBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    const auto& grad_output = tensor_args.grad_output;

    operation::Hash hash = operation::hash_operation<BinaryBackwardDeviceOperation>(
        static_cast<uint32_t>(args.op_type),
        args.output_dtype,
        args.output_memory_config,
        args.are_required_outputs,
        grad_output.dtype(),
        grad_output.memory_config(),
        input.dtype(),
        input.memory_config(),
        other.dtype(),
        other.memory_config(),
        input.padded_shape().volume());

    // Same reason as tanh_bw: when caller supplies its own output, that is what the
    // factory binds; dtype/layout/mem_config reach the kernels via CB formats and
    // TensorAccessorArgs, neither refreshable on a cache hit.
    const auto mix_preallocated = [&](const std::optional<Tensor>& preallocated) {
        if (preallocated.has_value()) {
            hash = ttsl::hash::hash_objects(
                hash, preallocated->dtype(), preallocated->layout(), preallocated->memory_config());
        }
    };
    mix_preallocated(tensor_args.preallocated_input_grad);
    mix_preallocated(tensor_args.preallocated_other_grad);

    return hash;
}

std::vector<Tensor> launch_binary_backward(
    BinaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& other,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    std::array<bool, 2> are_required_outputs,
    const std::optional<Tensor>& preallocated_input_grad,
    const std::optional<Tensor>& preallocated_other_grad) {
    auto operation_attributes = BinaryBackwardDeviceOperation::operation_attributes_t{
        .op_type = op_type,
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
        .are_required_outputs = are_required_outputs,
    };
    auto tensor_args = BinaryBackwardDeviceOperation::tensor_args_t{
        .grad_output = grad_output,
        .input = input,
        .other = other,
        .preallocated_input_grad = preallocated_input_grad,
        .preallocated_other_grad = preallocated_other_grad,
    };
    return ttnn::device_operation::launch<BinaryBackwardDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::binary_backward
