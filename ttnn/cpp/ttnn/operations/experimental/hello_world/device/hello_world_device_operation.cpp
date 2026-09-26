// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hello_world_device_operation.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

HelloWorldDeviceOperation::program_factory_t HelloWorldDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    log_debug(tt::LogOp, "[hello_world] select_program_factory: using HelloWorldProgramFactory");
    return HelloWorldProgramFactory{};
}

void HelloWorldDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    log_debug(
        tt::LogOp,
        "[hello_world] validate_on_program_cache_miss: input shape={}, dtype={}",
        input.padded_shape(),
        static_cast<int>(input.dtype()));

    TT_FATAL(
        input.layout() == Layout::TILE,
        "hello_world: input must have TILE layout, got {}",
        static_cast<int>(input.layout()));
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "hello_world: input must be DRAM interleaved, got memory layout {}",
        static_cast<int>(input.memory_config().memory_layout()));
}

HelloWorldDeviceOperation::spec_return_value_t HelloWorldDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    log_debug(tt::LogOp, "[hello_world] compute_output_specs: output spec mirrors input spec");
    // Identity op: the output has exactly the input's spec.
    return input.tensor_spec();
}

HelloWorldDeviceOperation::tensor_return_value_t HelloWorldDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    log_debug(tt::LogOp, "[hello_world] create_output_tensors: allocating output buffer on device");
    const auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor hello_world(const ttnn::Tensor& input_tensor) {
    using OperationType = ttnn::experimental::prim::HelloWorldDeviceOperation;
    log_debug(
        tt::LogOp,
        "[hello_world] launch: input shape={}, dtype={}",
        input_tensor.padded_shape(),
        static_cast<int>(input_tensor.dtype()));

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
