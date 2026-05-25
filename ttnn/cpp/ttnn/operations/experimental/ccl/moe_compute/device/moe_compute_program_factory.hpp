// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_compute_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct MoEComputeMeshWorkloadFactory {
    // Contract-2: declarative WorkloadDescriptor.  Per coord, builds a
    // ProgramDescriptor containing the tilize, matmul, and combine stages plus
    // their CBs and semaphores.  Workload-scoped init / final barrier
    // GlobalSemaphores are allocated once on cache miss and parked on
    // workload_descriptor.semaphores.  Tensor base addresses are bound via
    // emplace_runtime_args(Buffer*) -- and dynamic CBs are pinned via
    // CBDescriptor::buffer -- so the framework's fast cache-hit path patches
    // them on every dispatch.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MoEComputeParams& args,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores);

}  // namespace ttnn::experimental::prim
