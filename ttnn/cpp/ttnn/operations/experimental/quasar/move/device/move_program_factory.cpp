// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation_types.hpp"

namespace ttnn::prim::qsr {

tt::tt_metal::ProgramDescriptor MoveProgramFactory::create_descriptor(
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;
    using copy_attrs_t = CopyDeviceOperation::operation_attributes_t;
    using copy_args_t = CopyDeviceOperation::tensor_args_t;

    const copy_attrs_t copy_attrs{
        operation_attributes.output_mem_config, output.dtype(), operation_attributes.backwards};
    const copy_args_t copy_args{input, std::make_optional(output)};

    return CopyDeviceOperation::SameMemoryConfig::create_descriptor(copy_attrs, copy_args, output);
}

}  // namespace ttnn::prim::qsr
