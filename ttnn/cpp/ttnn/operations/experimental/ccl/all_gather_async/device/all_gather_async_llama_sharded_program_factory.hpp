// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct LlamaShardedMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index/forward/backward neighbours vary
    // across the mesh).
    //
    // GlobalSemaphores (semaphore[0], barrier_semaphore) live on
    // AllGatherAsyncParams — caller-allocated, so no workload-scoped semaphore
    // allocation is needed here.  The absolute addresses are written into the
    // writer runtime args every dispatch via the normal slow-path rebuild (the
    // framework re-calls create_workload_descriptor on cache hit when there
    // are no Buffer* bindings to patch).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
