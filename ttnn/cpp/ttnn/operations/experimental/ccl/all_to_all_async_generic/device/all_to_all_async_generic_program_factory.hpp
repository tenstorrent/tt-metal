// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_generic_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct AllToAllAsyncGenericProgram {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (device_index, forward/backward neighbours,
    // and per-link work distribution vary across the mesh).
    //
    // The init/final barrier GlobalSemaphores are workload-scoped: allocated
    // once in create_workload_descriptor on cache miss and parked in
    // wd.semaphores so they outlive the cached MeshWorkload.  Addresses are
    // baked into runtime args every dispatch via the slow-path rebuild (the
    // framework re-calls create_workload_descriptor on cache hit when there
    // are no Buffer* bindings to patch).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllAsyncGenericParams& operation_attributes,
        const AllToAllAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
