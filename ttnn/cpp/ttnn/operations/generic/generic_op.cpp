// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

namespace ttnn {

namespace {

// Runs the same program on every device of the tensors' mesh (SPMD).
tt::tt_metal::experimental::MeshProgramDescriptor make_spmd_mesh_program_descriptor(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);
    return mesh_program_descriptor;
}

}  // namespace

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) {
    return ttnn::prim::generic_op(io_tensors, mesh_program_descriptor);
}

Tensor generic_op(const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    return generic_op(io_tensors, make_spmd_mesh_program_descriptor(io_tensors, program_descriptor));
}

namespace experimental {

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) {
    ttnn::prim::prepare_generic_op(io_tensors, mesh_program_descriptor);
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    prepare_generic_op(io_tensors, make_spmd_mesh_program_descriptor(io_tensors, program_descriptor));
}

}  // namespace experimental

}  // namespace ttnn
