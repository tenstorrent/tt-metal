// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

namespace ttnn::operations::generic::program {

// generic_op is the meta-op that exposes ProgramDescriptor / MeshProgramDescriptor
// directly to users: the operation's "attributes" already ARE the descriptors.
// The factory's job is therefore to package the user-provided per-coord
// ProgramDescriptors into a WorkloadDescriptor.  There are no kernels, CBs or
// runtime args to synthesize here; cache-hit patching is handled by the
// framework via apply_resolved_bindings using whatever buffer bindings the user
// embedded in their ProgramDescriptor.
struct GenericMeshProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::generic::program
