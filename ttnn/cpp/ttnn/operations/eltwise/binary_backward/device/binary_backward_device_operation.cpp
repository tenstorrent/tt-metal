// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_device_operation.hpp"

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "binary_backward_op_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary_backward {

namespace {

// Binary backward ops accept only floating-point dtypes; int/uint operands are
// rejected regardless of what the other side carries, matching gelu_bw / tanh_bw.
bool is_supported_dtype(DataType dtype) {
    return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT8_B ||
           dtype == DataType::BFLOAT4_B;
}

std::string_view op_name_of(BinaryBackwardOpType op_type) {
    switch (op_type) {
        case BinaryBackwardOpType::MUL_BW: return "MUL_BW";
    }
    return "BINARY_BW";
}

// Hard: caller errors no path can recover from (bad storage, null buffer).
std::optional<std::string> operand_hard_reject_reason(
    std::string_view op_name, const Tensor& tensor, std::string_view role) {
    if (tensor.storage_type() != StorageType::DEVICE) {
        return fmt::format(
            "{} operation requires {} to be on Device, got storage_type={}", op_name, role, tensor.storage_type());
    }
    if (tensor.buffer() == nullptr) {
        return fmt::format(
            "{} operation requires {} to be allocated in a buffer on the device; buffer is null", op_name, role);
    }
    return std::nullopt;
}

// Soft: constraints the device op enforces but ttnn::multiply accepts (int/uint via
// mul_int_tile, ROW_MAJOR, non-32x32 tiles) — so route back to composite instead of fatalling.
std::optional<std::string> operand_soft_reject_reason(
    std::string_view op_name, const Tensor& tensor, std::string_view role) {
    if (!is_supported_dtype(tensor.dtype())) {
        return fmt::format(
            "{} device op only supports floating-point dtypes (bfloat16, float32, bfloat8_b, bfloat4_b); {} has "
            "dtype {}",
            op_name,
            role,
            tensor.dtype());
    }
    if (tensor.layout() != Layout::TILE) {
        return fmt::format("{} device op requires {} in TILE layout, got {}", op_name, role, tensor.layout());
    }
    const auto tile = tensor.tensor_spec().tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
        return fmt::format(
            "{} device op only supports 32x32 tiles, but {} has a {}x{} tile",
            op_name,
            role,
            tile.get_height(),
            tile.get_width());
    }
    return std::nullopt;
}

// Preallocated dtype must match its operand (pack CB format is fixed to the buffer's dtype),
// and shape must match (writer offsets are keyed off input.physical_volume()).
std::optional<std::string> preallocated_hard_reject_reason(
    std::string_view op_name,
    const std::optional<Tensor>& preallocated,
    const Tensor& reference,
    const Tensor& input,
    std::string_view role) {
    if (!preallocated.has_value()) {
        return std::nullopt;
    }
    if (auto r = operand_hard_reject_reason(op_name, *preallocated, role); r.has_value()) {
        return r;
    }
    if (preallocated->device() != input.device()) {
        return fmt::format("{} operation requires {} to be on the same device as the input", op_name, role);
    }
    if (preallocated->dtype() != reference.dtype()) {
        return fmt::format(
            "{} operation requires {} dtype to match its operand, got {} vs {}",
            op_name,
            role,
            preallocated->dtype(),
            reference.dtype());
    }
    if (preallocated->logical_shape() != reference.logical_shape()) {
        return fmt::format(
            "{} operation requires {} logical shape to match its operand, got {} vs {}",
            op_name,
            role,
            preallocated->logical_shape(),
            reference.logical_shape());
    }
    if (preallocated->padded_shape() != reference.padded_shape()) {
        return fmt::format(
            "{} operation requires {} padded shape to match its operand, got {} vs {}",
            op_name,
            role,
            preallocated->padded_shape(),
            reference.padded_shape());
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> BinaryBackwardDeviceOperation::hard_invariants_reason(
    BinaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& other,
    const std::optional<Tensor>& preallocated_input_grad,
    const std::optional<Tensor>& preallocated_other_grad) {
    const auto op_name = op_name_of(op_type);

    if (auto r = operand_hard_reject_reason(op_name, grad_output, "the grad_output tensor"); r.has_value()) {
        return r;
    }
    if (auto r = operand_hard_reject_reason(op_name, input, "the input tensor"); r.has_value()) {
        return r;
    }
    if (auto r = operand_hard_reject_reason(op_name, other, "the other tensor"); r.has_value()) {
        return r;
    }
    if (!(grad_output.device() == input.device() && other.device() == input.device())) {
        return fmt::format("{} operation requires all operands on the same device", op_name);
    }

    if (auto r = preallocated_hard_reject_reason(
            op_name, preallocated_input_grad, input, input, "the preallocated input_grad tensor");
        r.has_value()) {
        return r;
    }
    if (auto r = preallocated_hard_reject_reason(
            op_name, preallocated_other_grad, other, input, "the preallocated other_grad tensor");
        r.has_value()) {
        return r;
    }

    return std::nullopt;
}

std::optional<std::string> BinaryBackwardDeviceOperation::soft_fallback_reason(
    BinaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_memory_config,
    std::array<bool, 2> are_required_outputs,
    const std::optional<Tensor>& preallocated_input_grad,
    const std::optional<Tensor>& preallocated_other_grad) {
    const auto op_name = op_name_of(op_type);

    // MUL_BW's first-cut contract: both grads must be requested. Partial-mask callers
    // fall back to the composite path in binary_backward.cpp.
    if (!(are_required_outputs[0] && are_required_outputs[1])) {
        return fmt::format(
            "{} operation currently requires both grads; got are_required_outputs=[{}, {}]",
            op_name,
            are_required_outputs[0],
            are_required_outputs[1]);
    }

    // Per-operand + per-preallocated soft rejects (int/uint dtype, non-TILE, non-32x32);
    // ttnn::multiply accepts all of these, so route back to composite instead of fatalling.
    for (const auto& [tensor, role] :
         {std::pair{std::cref(grad_output), "the grad_output tensor"},
          std::pair{std::cref(input), "the input tensor"},
          std::pair{std::cref(other), "the other tensor"}}) {
        if (auto r = operand_soft_reject_reason(op_name, tensor.get(), role); r.has_value()) {
            return r;
        }
    }
    for (const auto& [preallocated, role] :
         {std::pair{std::cref(preallocated_input_grad), "the preallocated input_grad tensor"},
          std::pair{std::cref(preallocated_other_grad), "the preallocated other_grad tensor"}}) {
        if (preallocated.get().has_value()) {
            if (auto r = operand_soft_reject_reason(op_name, *preallocated.get(), role); r.has_value()) {
                return r;
            }
        }
    }

    // Sharded inputs + sharded preallocated grads: fused device op is interleaved-only,
    // and compute_output_specs returns preallocated->tensor_spec() verbatim, so a sharded
    // preallocated would slip past the input-only gate and be written by an interleaved kernel.
    for (const auto& [tensor, role] :
         {std::pair{std::cref(grad_output), "the grad_output tensor"},
          std::pair{std::cref(input), "the input tensor"},
          std::pair{std::cref(other), "the other tensor"}}) {
        if (tensor.get().is_sharded()) {
            return fmt::format(
                "{} operation does not support sharded {} (memory_layout={})",
                op_name,
                role,
                tensor.get().memory_config().memory_layout());
        }
    }
    for (const auto& [preallocated, role] :
         {std::pair{std::cref(preallocated_input_grad), "the preallocated input_grad tensor"},
          std::pair{std::cref(preallocated_other_grad), "the preallocated other_grad tensor"}}) {
        if (preallocated.get().has_value() && preallocated.get()->is_sharded()) {
            return fmt::format(
                "{} operation does not support sharded {} (memory_layout={})",
                op_name,
                role,
                preallocated.get()->memory_config().memory_layout());
        }
    }

    // logical_shape catches TILE-padding equalising padded_shape while broadcast semantics
    // remain (e.g. (1,1,32,128) vs (1,1,1,128) both pad to (1,1,32,128)).
    if (!(grad_output.logical_shape() == input.logical_shape() && other.logical_shape() == input.logical_shape())) {
        return fmt::format(
            "{} operation requires grad_output, input, and other to have the same logical shape, got {}, {}, {}",
            op_name,
            grad_output.logical_shape(),
            input.logical_shape(),
            other.logical_shape());
    }
    if (!(grad_output.padded_shape() == input.padded_shape() && other.padded_shape() == input.padded_shape())) {
        return fmt::format(
            "{} operation requires grad_output, input, and other to have the same padded shape, got {}, {}, {}",
            op_name,
            grad_output.padded_shape(),
            input.padded_shape(),
            other.padded_shape());
    }

    // A1: with no explicit output_memory_config, keep the call on the composite when input
    // and other differ so other_grad isn't silently placed at input's mem_config.
    if (!output_memory_config.has_value() && !(input.memory_config() == other.memory_config())) {
        return fmt::format(
            "{} operation with no output_memory_config requires input and other to share a memory_config so both "
            "grads land where the composite path would place them, got input={} vs other={}",
            op_name,
            input.memory_config(),
            other.memory_config());
    }

    // Per-output effective memory layout: preallocated wins (compute_output_specs returns
    // preallocated->tensor_spec() verbatim), else the resolved output_memory_config.
    const auto& resolved_mem_config = output_memory_config.value_or(input.memory_config());
    const auto effective_mem_layout = [&](const std::optional<Tensor>& preallocated) {
        return preallocated.has_value() ? preallocated->memory_config().memory_layout()
                                        : resolved_mem_config.memory_layout();
    };
    if (effective_mem_layout(preallocated_input_grad) != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format(
            "{} operation requires INTERLEAVED memory layout for input_grad, got {}",
            op_name,
            effective_mem_layout(preallocated_input_grad));
    }
    if (effective_mem_layout(preallocated_other_grad) != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format(
            "{} operation requires INTERLEAVED memory layout for other_grad, got {}",
            op_name,
            effective_mem_layout(preallocated_other_grad));
    }

    return std::nullopt;
}

void BinaryBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // args.output_memory_config is always resolved by the time the launch reaches here, so
    // pass it as an engaged optional; the A1 default-parity rule is inert in that case.
    auto hard = hard_invariants_reason(
        args.op_type,
        tensor_args.grad_output,
        tensor_args.input,
        tensor_args.other,
        tensor_args.preallocated_input_grad,
        tensor_args.preallocated_other_grad);
    TT_FATAL(!hard.has_value(), "{}", hard.value_or(std::string{}));
    auto soft = soft_fallback_reason(
        args.op_type,
        tensor_args.grad_output,
        tensor_args.input,
        tensor_args.other,
        args.output_memory_config,
        args.are_required_outputs,
        tensor_args.preallocated_input_grad,
        tensor_args.preallocated_other_grad);
    TT_FATAL(!soft.has_value(), "{}", soft.value_or(std::string{}));
}

BinaryBackwardDeviceOperation::spec_return_value_t BinaryBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Carry the operand's padded_shape through so an input padded beyond tile alignment
    // (e.g. tilize_with_val_padding: logical 40x40 in padded 96x96 = 9 tiles) gets an
    // output buffer sized for the pages the writer emits, not the tile-padded logical.
    const auto spec_for = [&](const Tensor& ref, const std::optional<Tensor>& preallocated) -> TensorSpec {
        if (preallocated.has_value()) {
            return preallocated->tensor_spec();
        }
        DataType output_dtype = args.output_dtype;
        if (output_dtype == DataType::INVALID) {
            output_dtype = ref.dtype();
        }
        return TensorSpec(
            ref.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::TILE),
                args.output_memory_config,
                ref.logical_shape(),
                ref.padded_shape()));
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

    // Preallocated dtype/layout/mem_config reach the kernels via CB formats and
    // TensorAccessorArgs, neither refreshable on a cache hit (same as tanh_bw).
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
