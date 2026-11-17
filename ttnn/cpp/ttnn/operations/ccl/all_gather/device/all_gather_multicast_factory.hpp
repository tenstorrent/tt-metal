// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"

#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::ccl {

struct AllGatherMulticastFactory {
    // Workload-scoped semaphores; one ProgramDescriptor per coord; tensor addresses are Buffer* bindings.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherParams& operation_attributes,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::ccl
