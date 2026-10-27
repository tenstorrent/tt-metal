// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation_types.hpp"

namespace ttnn::prim {

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

void MoveProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Delegates to CopyDeviceOperation::SameMemoryConfig, whose reader/writer take their
    // buffer at slot 0; the sharding tail args are shape-derived and therefore keyed.
    const uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, 0)) {
        for (auto& a : col) {
            if (a.size() > 0) {
                a[0] = src_addr;
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, 1)) {
        for (auto& a : col) {
            if (a.size() > 0) {
                a[0] = dst_addr;
            }
        }
    }
}

}  // namespace ttnn::prim
