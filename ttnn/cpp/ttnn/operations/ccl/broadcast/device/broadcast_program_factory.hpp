// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/broadcast/device/broadcast_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"

#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::prim {

struct BroadcastProgramFactory {
    // Workload-scoped semaphores; one ProgramDescriptor per coord; tensor addresses are Buffer* bindings.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const BroadcastParams& operation_attributes,
        const BroadcastInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
