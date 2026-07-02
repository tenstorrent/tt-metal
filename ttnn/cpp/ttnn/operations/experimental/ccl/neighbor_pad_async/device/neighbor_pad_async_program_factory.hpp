// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct NeighborPadAsyncMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (forward / backward neighbour availability,
    // device index, etc. vary across the mesh).
    //
    // The GlobalSemaphores (h, w, barrier) live on NeighborPadAsyncParams
    // (caller allocated) so the factory needs no workload-scoped resources.
    // Buffer base addresses (input/output) are wired up via Buffer* runtime
    // args so the framework patches them on every dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
