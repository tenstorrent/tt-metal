// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "dispatch_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

struct DispatchProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 2>;

    // Declarative WorkloadDescriptor entry point (Contract 2).  Allocates the
    // three GlobalSemaphores once per cache miss (parked in
    // `WorkloadDescriptor::semaphores` so their device-side allocations outlive
    // the cached workload), runs the cross-device Synchronize barrier, then
    // builds one ProgramDescriptor per mesh coordinate.  The framework
    // realises this into the cached MeshWorkload.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DispatchParams& operation_attributes,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
