// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::qsr {

void RoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::TensorMemoryLayout;

    const auto check_operand = [](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == ttnn::StorageType::DEVICE, "routed_expert_ffn: {} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "routed_expert_ffn: {} must be allocated on device", name);
        TT_FATAL(t.layout() == Layout::TILE, "routed_expert_ffn: {} must be TILE layout, got {}", name, t.layout());
        TT_FATAL(t.dtype() == DataType::BFLOAT16, "routed_expert_ffn: {} must be BFLOAT16, got {}", name, t.dtype());
        TT_FATAL(
            t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                t.memory_config().buffer_type() == BufferType::DRAM,
            "routed_expert_ffn: {} must be DRAM interleaved, got {}",
            name,
            t.memory_config());
        TT_FATAL(
            t.padded_shape().volume() == t.padded_shape()[-2] * t.padded_shape()[-1],
            "routed_expert_ffn: {} must be a 2D matrix (all outer dims 1), got {}",
            name,
            t.padded_shape());
    };
    const auto& x = tensor_args.x;
    const auto& w_gate = tensor_args.w_gate;
    const auto& w_up = tensor_args.w_up;
    const auto& w_down = tensor_args.w_down;
    check_operand(x, "x");
    check_operand(w_gate, "w_gate");
    check_operand(w_up, "w_up");
    check_operand(w_down, "w_down");

    const uint32_t k = x.padded_shape()[-1];
    const uint32_t h = w_gate.padded_shape()[-1];
    TT_FATAL(
        w_gate.padded_shape()[-2] == k && w_up.padded_shape()[-2] == k && w_up.padded_shape()[-1] == h,
        "routed_expert_ffn: w_gate and w_up must be (K, H) = ({}, {}), got {} and {}",
        k,
        h,
        w_gate.padded_shape(),
        w_up.padded_shape());
    TT_FATAL(
        w_down.padded_shape()[-2] == h && w_down.padded_shape()[-1] == k,
        "routed_expert_ffn: w_down must be (H, K) = ({}, {}), got {}",
        h,
        k,
        w_down.padded_shape());
    for (const Tensor* t : {&w_gate, &w_up, &w_down}) {
        TT_FATAL(t->device() == x.device(), "routed_expert_ffn: all inputs must be on the same device");
    }
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            args.output_mem_config.buffer_type() == BufferType::DRAM,
        "routed_expert_ffn: output must be DRAM interleaved, got {}",
        args.output_mem_config);
}

RoutedExpertFfnDeviceOperation::spec_return_value_t RoutedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x;
    return tt::tt_metal::TensorSpec(
        x.logical_shape(),
        tt::tt_metal::TensorLayout(
            x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

RoutedExpertFfnDeviceOperation::tensor_return_value_t RoutedExpertFfnDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.x.device());
}

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = RoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.x = x, .w_gate = w_gate, .w_up = w_up, .w_down = w_down});
}

}  // namespace ttnn::prim::qsr
