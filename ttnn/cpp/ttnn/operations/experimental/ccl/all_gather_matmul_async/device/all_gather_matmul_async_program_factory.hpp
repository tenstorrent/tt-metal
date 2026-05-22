// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

// Contract-2 (descriptor) factory for the fused AllGather + Matmul op.  All
// per-coord program building happens through the ProgramDescriptor variants of
// the matmul (1D/2D mcast) and AllGather minimal builders; the workload-scoped
// GlobalSemaphores live on the AllGatherAsyncParams that the caller provided
// (no extra ones are needed).
struct AllGatherMatmulAsyncMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherMatmulAsyncParams& operation_attributes,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
