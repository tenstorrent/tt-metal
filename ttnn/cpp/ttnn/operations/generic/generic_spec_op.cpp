// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_spec_op.hpp"

#include "device/generic_spec_op_device_operation.hpp"

namespace ttnn {

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::ProgramSpec& program_spec,
    const ProgramSpecRunArgs& run_args) {
    using tt::tt_metal::experimental::TensorParamName;
    TT_FATAL(!io_tensors.empty(), "generic_op: io_tensors must contain at least the output tensor");

    ttnn::operations::generic::GenericSpecOpDeviceOperation::operation_attributes_t attributes{
        .spec = program_spec,
        .kernel_run_args = run_args.kernel_run_args,
        .custom_program_hash = run_args.custom_program_hash,
    };
    // Tensor copies share one MeshTensor, so identity finds the tensor's slot in io_tensors.
    for (const auto& [tensor_parameter_name, tensor] : run_args.tensor_args) {
        const auto* target = &tensor.mesh_tensor();
        std::size_t io_index = 0;
        while (io_index < io_tensors.size() && &io_tensors[io_index].mesh_tensor() != target) {
            ++io_index;
        }
        TT_FATAL(
            io_index < io_tensors.size(),
            "generic_op: the tensor bound to TensorParameter '{}' is not in io_tensors",
            tensor_parameter_name);
        attributes.tensor_bindings.emplace_back(TensorParamName{tensor_parameter_name}, io_index);
    }
    return ttnn::prim::generic_spec_op(io_tensors, attributes);
}

}  // namespace ttnn
