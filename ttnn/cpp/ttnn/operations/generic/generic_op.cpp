// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn::operations::generic {

Tensor GenericOp::invoke(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) {
    return ttnn::prim::generic_op(io_tensors, mesh_program_descriptor);
}

Tensor GenericOp::invoke(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    // Create SPMD MeshProgramDescriptor; same program for the entire mesh
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    return invoke(io_tensors, mesh_program_descriptor);
}

}  // namespace ttnn::operations::generic
