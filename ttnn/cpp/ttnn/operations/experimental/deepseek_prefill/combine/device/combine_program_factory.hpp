// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "combine_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine {

struct CombineProgramFactory {
    // Declarative WorkloadDescriptor entry point (Contract 2).  Allocates the two
    // GlobalSemaphores once per cache miss (parked in `WorkloadDescriptor::semaphores`
    // so their device-side allocations outlive the cached workload), runs the
    // cross-device Synchronize barrier, then builds one ProgramDescriptor per
    // mesh coordinate.  The framework realises this into the cached MeshWorkload.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineParams& operation_attributes,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
