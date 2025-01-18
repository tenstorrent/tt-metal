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

GenericOpDeviceOperation::shape_return_value_t GenericOpDeviceOperation::compute_output_shapes(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.io_tensors.back().shape();
}

GenericOpDeviceOperation::tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<GenericOpDeviceOperation::operation_attributes_t, GenericOpDeviceOperation::tensor_args_t>
GenericOpDeviceOperation::invoke(
    const Tensor& input,
    const GenericOpDeviceOperation::operation_attributes_t& operation_attributes,
    const std::vector<Tensor>& io_tensors) {
    return {
        operation_attributes_t{
            .circular_buffer_attributes = cb_attr_map,
            .data_movement_attributes = dm_attr,
            .compute_attributes = comp_attr},
        tensor_args_t{.input_tensor = input, .io_tensors = io_tensors}};
}

}  // namespace ttnn::operations::generic
