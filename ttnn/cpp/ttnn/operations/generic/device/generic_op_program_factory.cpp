// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_program_factory.hpp"

namespace ttnn::operations::generic::program {

using namespace tt::tt_metal;

tt::tt_metal::WorkloadDescriptor GenericMeshProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    // The user-supplied MeshProgramDescriptor (== operation_attributes) already
    // contains one ProgramDescriptor per MeshCoordinateRange. The factory has
    // no workload-scoped resources of its own -- it just forwards each
    // (range, descriptor) pair into the framework's WorkloadDescriptor so that
    // the adapter can materialise each one into a Program on cache miss and
    // patch buffer addresses on cache hits.
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.programs.reserve(operation_attributes.mesh_programs.size());
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        workload_descriptor.programs.push_back({mesh_coord_range, program_descriptor});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::generic::program
