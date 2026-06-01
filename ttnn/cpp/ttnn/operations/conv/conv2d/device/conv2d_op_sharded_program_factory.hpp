// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Satisfies `ProgramDescriptorFactoryConcept` via the declarative
// `create_workload_descriptor` contract (contract 2 in mesh_device_operation_adapter.hpp).
struct Conv2dShardedProgramFactory {
    // Builds the entire workload in one call (invoked ONCE per workload on
    // cache miss):
    // On cache hits the framework's `BufferBinding` fast path patches the
    // weight / bias buffer addresses recorded via
    // `KernelDescriptor::emplace_runtime_args(Buffer*)` and the dynamic CB
    // addresses recorded on `CBDescriptor::buffer` — `create_workload_descriptor`
    // is not re-invoked.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
