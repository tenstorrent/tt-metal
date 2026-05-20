// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct AllToAllAsyncProgram {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index, forward/backward neighbours,
    // strides/offsets, and the receiver-side mimicking of each remote sender
    // vary across the mesh).
    //
    // The single GlobalSemaphore lives on AllToAllAsyncParams (caller
    // allocated) so this factory needs no workload-scoped resources — the
    // semaphore address is workload-scoped (stable across dispatches) and
    // written as a raw uint32_t.  Tensor buffer addresses (input, persistent
    // intermediate/output) are patched on cache hit via BufferBindings
    // (emplace_runtime_args); there is no slow-path rebuild in contract (2).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllAsyncParams& operation_attributes,
        const AllToAllAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
