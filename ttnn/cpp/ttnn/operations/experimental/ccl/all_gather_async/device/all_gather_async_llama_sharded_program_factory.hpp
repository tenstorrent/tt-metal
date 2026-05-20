// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct LlamaShardedMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index/forward/backward neighbours vary
    // across the mesh).
    //
    // GlobalSemaphores (semaphore[0], barrier_semaphore) live on
    // AllGatherAsyncParams — caller-allocated, so no workload-scoped semaphore
    // allocation is needed here.  Tensor buffer addresses (input/output) are
    // patched on cache hit via BufferBindings (emplace_runtime_args); semaphore
    // addresses are stable across dispatches and written as raw uint32_t.
    // Contract (2) has no slow-path rebuild on cache hit.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
