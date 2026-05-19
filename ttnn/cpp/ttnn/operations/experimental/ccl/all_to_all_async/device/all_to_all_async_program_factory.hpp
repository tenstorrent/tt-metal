// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct AllToAllAsyncProgram {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (ring_index, forward/backward neighbours,
    // strides/offsets, and the receiver-side mimicking of each remote sender
    // vary across the mesh).
    //
    // The single GlobalSemaphore lives on AllToAllAsyncParams (caller
    // allocated) so this factory needs no workload-scoped resources — the
    // absolute address is written into the writer/receiver runtime args every
    // dispatch via the normal slow-path rebuild (the framework re-calls
    // create_workload_descriptor on cache hit when there are no Buffer*
    // bindings to patch).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllAsyncParams& operation_attributes,
        const AllToAllAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
