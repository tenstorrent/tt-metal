// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "combine_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

struct CombineFabric2dProgramFactory {
    // Contract-2 declarative WorkloadDescriptor entry point. Allocates workload-scope
    // GlobalSemaphores (the receiver data-ready semaphore, and in later phases the producer
    // credit semaphore) once per cache miss so their device-side addresses are uniform across
    // the mesh, then builds one ProgramDescriptor per mesh coordinate (each chip sends to its
    // own neighbor, so compile-time args are coord-dependent and cannot be replicated).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
