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
    // The user-provided MeshProgramDescriptor already pairs MeshCoordinateRanges
    // with ProgramDescriptors; simply forward those entries into the workload.
    // No workload-scoped buffers/semaphores are owned here because the user
    // builds and owns any intermediate resources referenced by their kernels.
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.programs.reserve(operation_attributes.mesh_programs.size());
    for (const auto& [range, descriptor] : operation_attributes.mesh_programs) {
        workload_descriptor.programs.push_back({range, descriptor});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::generic::program
