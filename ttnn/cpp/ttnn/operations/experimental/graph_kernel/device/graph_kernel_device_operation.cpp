// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_kernel_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

void GraphKernelDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& inputs = tensor_args.inputs;
    TT_FATAL(!inputs.empty(), "graph_kernel: at least one input tensor is required");

    const auto& first = inputs.front();
    TT_FATAL(first.storage_type() == StorageType::DEVICE, "graph_kernel: inputs must be on device");
    auto* device = first.device();

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& t = inputs[i];
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "graph_kernel: input {} must be on device", i);
        TT_FATAL(t.buffer() != nullptr, "graph_kernel: input {} must be allocated on device", i);
        TT_FATAL(t.device() == device, "graph_kernel: input {} must be on the same device as input 0", i);
        TT_FATAL(
            t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "graph_kernel: input {} must be interleaved (sharded inputs are not supported yet)",
            i);
    }
}

GraphKernelDeviceOperation::spec_return_value_t GraphKernelDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Basis: the output mirrors inputs[0].
    return tensor_args.inputs.front().tensor_spec();
}

GraphKernelDeviceOperation::tensor_return_value_t GraphKernelDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.inputs.front().device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor graph_kernel(const std::vector<Tensor>& inputs, const std::string& text) {
    using OperationType = ttnn::experimental::prim::GraphKernelDeviceOperation;

    // The launch path derives the target device from the first tensor in tensor_args
    // before validate_on_program_cache_miss runs, so an empty list must be rejected here.
    TT_FATAL(!inputs.empty(), "graph_kernel: at least one input tensor is required");

    auto operation_attributes = OperationType::operation_attributes_t{.text = text};
    auto tensor_args = OperationType::tensor_args_t{.inputs = inputs};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
