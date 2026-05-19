// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rms_allgather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct RMSAllGatherMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.  Builds one
    // ProgramDescriptor per coord.  The cross-device GlobalSemaphore lives
    // on operation_attributes (caller-allocated) so the factory needs no
    // workload-scoped resources.  All other semaphores are program-scoped
    // (reserved via SemaphoreDescriptor slots).  Dynamic CBs that point at
    // input/residual/output/stats tensor buffers are wired up via
    // CBDescriptor::buffer so the framework patches their addresses on
    // every dispatch.  Tensor base addresses in writer runtime args
    // (stats, gamma) and the cross-device GlobalSemaphore address are
    // refreshed via re-emission on cache hits.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
