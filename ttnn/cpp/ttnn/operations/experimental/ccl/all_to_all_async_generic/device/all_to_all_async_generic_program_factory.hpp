// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_generic_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct AllToAllAsyncGenericProgram {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (device_index, forward/backward neighbours,
    // and per-link work distribution vary across the mesh).
    //
    // The init/final barrier GlobalSemaphores are workload-scoped: allocated
    // once in create_workload_descriptor on cache miss and parked in
    // wd.semaphores so they outlive the cached MeshWorkload.  Their addresses
    // are stable across dispatches and written as raw uint32_t.  Tensor buffer
    // addresses (input + output) are patched on cache hit via BufferBindings
    // (emplace_runtime_args); there is no slow-path rebuild in contract (2).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllAsyncGenericParams& operation_attributes,
        const AllToAllAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
