// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

namespace ttnn::operations::generic::program {

// Declarative WorkloadDescriptor factory for the generic op.
//
// The generic op is a thin pass-through: the user constructs a
// MeshProgramDescriptor (one ProgramDescriptor per MeshCoordinateRange) and
// the factory simply forwards each (range, descriptor) pair into the
// framework-owned WorkloadDescriptor.
//
// The framework adapter (DescriptorMeshWorkloadAdapter, contract 2) realises
// each ProgramDescriptor into a Program once on cache miss and patches buffer
// addresses on cache hits via the BufferBinding fast path -- no manual
// override_runtime_arguments is required.
struct GenericMeshProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::generic::program
