// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_add_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::qsr {

void SimpleAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::TensorMemoryLayout;

    const auto check_operand = [](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == ttnn::StorageType::DEVICE, "simple_add: {} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "simple_add: {} must be allocated on device", name);
        TT_FATAL(t.layout() == Layout::TILE, "simple_add: {} must be TILE layout, got {}", name, t.layout());
        TT_FATAL(t.dtype() == DataType::BFLOAT16, "simple_add: {} must be BFLOAT16, got {}", name, t.dtype());
        TT_FATAL(
            t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                t.memory_config().buffer_type() == BufferType::DRAM,
            "simple_add: {} must be DRAM interleaved, got {}",
            name,
            t.memory_config());
    };
    const auto& a = tensor_args.input_a;
    const auto& b = tensor_args.input_b;
    check_operand(a, "input_a");
    check_operand(b, "input_b");
    TT_FATAL(
        a.padded_shape() == b.padded_shape(),
        "simple_add: inputs must have the same shape (no broadcast), got {} and {}",
        a.padded_shape(),
        b.padded_shape());
    TT_FATAL(a.device() == b.device(), "simple_add: inputs must be on the same device");
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            args.output_mem_config.buffer_type() == BufferType::DRAM,
        "simple_add: output must be DRAM interleaved, got {}",
        args.output_mem_config);
}

SimpleAddDeviceOperation::spec_return_value_t SimpleAddDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_a;
    return tt::tt_metal::TensorSpec(
        a.logical_shape(),
        tt::tt_metal::TensorLayout(
            a.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

SimpleAddDeviceOperation::tensor_return_value_t SimpleAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input_a.device());
}

ttnn::Tensor simple_add(
    const ttnn::Tensor& input_a, const ttnn::Tensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = SimpleAddDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input_a = input_a, .input_b = input_b});
}

}  // namespace ttnn::prim::qsr
