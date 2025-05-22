// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

GenericOpDeviceOperation::program_factory_t GenericOpDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return GenericProgram{};
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

GenericOpDeviceOperation::spec_return_value_t GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.output_tensor.get_tensor_spec();
}

GenericOpDeviceOperation::tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    return tensor_args.output_tensor;
}

std::tuple<GenericOpDeviceOperation::operation_attributes_t, GenericOpDeviceOperation::tensor_args_t>
GenericOpDeviceOperation::invoke(
    const std::vector<Tensor>& io_tensors, const operation_attributes_t& operation_attributes) {
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());

    // NOTE: The output tensor is the last one in the vector, the rest are input tensors
    // Passing in output_tensors into tensor_args_t like this for clarity reasons.
    return {operation_attributes, tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()}};
}

}  // namespace ttnn::operations::generic
