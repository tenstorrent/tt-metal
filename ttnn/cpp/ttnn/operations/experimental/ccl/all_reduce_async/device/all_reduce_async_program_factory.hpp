// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct AllReduceAsyncMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord (device_index / forward & backward neighbours
    // vary across the mesh).
    //
    // The single GlobalSemaphore lives on AllReduceAsyncParams (caller
    // allocated); per-link reduction semaphores are program-scoped and
    // expressed as SemaphoreDescriptors in the per-coord descriptor.  Two
    // dynamic CBs (cb_reduction over buffer_tensor, cb_out over output_tensor)
    // are wired through CBDescriptor::buffer so the framework patches their
    // addresses on every dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllReduceAsyncParams& operation_attributes,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores);

}  // namespace ttnn
