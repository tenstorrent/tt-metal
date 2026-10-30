// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"
#include "device/generic_program_spec_device_operation.hpp"

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

namespace {

using ProgramSpecAttributes = ttnn::operations::generic::GenericProgramSpecDeviceOperation::operation_attributes_t;

// No range: the program runs on every device that holds the tensors.
ProgramSpecAttributes make_program_spec_attributes(
    const ProgramSpec& spec, const ProgramRunArgs& run_args, std::optional<uint64_t> custom_program_hash) {
    return {.programs = {{.spec = &spec, .run_args = &run_args}}, .custom_program_hash = custom_program_hash};
}

// Pairs each range's spec with its run args.
ProgramSpecAttributes make_program_spec_attributes(
    const MeshProgramSpec& mesh_program_spec, const MeshProgramRunArgs& mesh_run_args) {
    const auto& programs = mesh_program_spec.programs;
    TT_FATAL(!programs.empty(), "generic_op: the MeshProgramSpec has no programs");
    TT_FATAL(
        mesh_run_args.programs.size() == programs.size(),
        "generic_op: the MeshProgramSpec has {} programs, but the MeshProgramRunArgs has run args for {}",
        programs.size(),
        mesh_run_args.programs.size());
    ProgramSpecAttributes attributes{.custom_program_hash = mesh_program_spec.custom_program_hash};
    attributes.programs.reserve(programs.size());
    for (const auto& [range, spec] : programs) {
        const auto run_args =
            std::ranges::find(mesh_run_args.programs, range, &std::pair<MeshCoordinateRange, ProgramRunArgs>::first);
        TT_FATAL(
            run_args != mesh_run_args.programs.end(),
            "generic_op: the MeshProgramRunArgs has no run args for mesh range {}",
            range);
        attributes.programs.push_back({.range = range, .spec = &spec, .run_args = &run_args->second});
    }
    return attributes;
}

}  // namespace

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const ProgramSpec& spec,
    const ProgramRunArgs& run_args,
    std::optional<uint64_t> custom_program_hash) {
    return ttnn::prim::generic_op(io_tensors, make_program_spec_attributes(spec, run_args, custom_program_hash));
}

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const MeshProgramSpec& mesh_program_spec,
    const MeshProgramRunArgs& mesh_run_args) {
    return ttnn::prim::generic_op(io_tensors, make_program_spec_attributes(mesh_program_spec, mesh_run_args));
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const ProgramSpec& spec,
    const ProgramRunArgs& run_args,
    std::optional<uint64_t> custom_program_hash) {
    ttnn::prim::prepare_generic_op(io_tensors, make_program_spec_attributes(spec, run_args, custom_program_hash));
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const MeshProgramSpec& mesh_program_spec,
    const MeshProgramRunArgs& mesh_run_args) {
    ttnn::prim::prepare_generic_op(io_tensors, make_program_spec_attributes(mesh_program_spec, mesh_run_args));
}

uint64_t compute_program_spec_hash(const ProgramSpec& spec) {
    return ttnn::operations::generic::compute_program_spec_hash(spec);
}

}  // namespace experimental

}  // namespace ttnn
