// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_program_factory.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation_types.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::prim {

MoveProgramFactory::cached_program_t MoveProgramFactory::create(
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    using copy_attrs_t = CopyDeviceOperation::operation_attributes_t;
    using copy_args_t = CopyDeviceOperation::tensor_args_t;

    const copy_attrs_t copy_attrs{
        operation_attributes.output_mem_config, output.dtype(), operation_attributes.backwards};
    const copy_args_t copy_args{input, std::make_optional(output)};

    return ttnn::prim::CopyProgramFactory::create(copy_attrs, copy_args, tensor_return_value);
}

void MoveProgramFactory::override_runtime_arguments(
    MoveProgramFactory::cached_program_t& cached_program,
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using copy_attrs_t = CopyDeviceOperation::operation_attributes_t;
    using copy_args_t = CopyDeviceOperation::tensor_args_t;
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    const copy_attrs_t copy_attrs{
        operation_attributes.output_mem_config, output.dtype(), operation_attributes.backwards};
    const copy_args_t copy_args{input, std::make_optional(output)};

    ttnn::prim::CopyProgramFactory::override_runtime_arguments(
        cached_program, copy_attrs, copy_args, tensor_return_value);
}

}  // namespace ttnn::prim
