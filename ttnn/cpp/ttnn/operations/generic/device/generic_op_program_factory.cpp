// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_program_factory.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::generic::program {

using namespace tt::tt_metal;

namespace {

// Build a placeholder descriptor for a mesh coordinate that no mesh_program covers.
// A MeshWorkload requires every program it contains to report identical per-core-type
// config sizes; An empty ProgramDescriptor{} finalizes to all-zero offsets; All
//  covered programs must already share identical config sizes for the workload to
// be valid.
tt::tt_metal::ProgramDescriptor make_config_size_matching_placeholder(
    const tt::tt_metal::ProgramDescriptor& reference) {
    return reference;
}

}  // namespace

tt::tt_metal::ProgramDescriptor GenericMeshDescriptorFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_programs = operation_attributes.mesh_programs;
    TT_FATAL(!mesh_programs.empty(), "generic_op: MeshProgramDescriptor.mesh_programs must not be empty");

    if (mesh_dispatch_coordinate.has_value()) {
        const auto& coord = mesh_dispatch_coordinate.value();
        for (const auto& [range, desc] : mesh_programs) {
            if (range.contains(coord)) {
                return desc;
            }
        }
        // No mesh_program covers this coordinate. Return a placeholder whose config-size footprint
        // matches the covered programs so the MeshWorkload's identical-config-size invariant holds.
        return make_config_size_matching_placeholder(mesh_programs.front().second);
    }

    TT_FATAL(
        mesh_programs.size() == 1,
        "generic_op: multiple mesh_program entries ({}) require per-coordinate mesh dispatch",
        mesh_programs.size());
    return mesh_programs.front().second;
}

}  // namespace ttnn::operations::generic::program
