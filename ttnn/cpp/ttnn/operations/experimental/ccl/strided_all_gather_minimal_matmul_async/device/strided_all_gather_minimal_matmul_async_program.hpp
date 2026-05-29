// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct StridedAllGatherMinimalMatmulAsyncProgramFactory {
    // Contract-2: declarative WorkloadDescriptor.  Per coord, builds a
    // ProgramDescriptor by composing the ProgramDescriptor variants of the
    // minimal_matmul helper and the strided all-gather builder.  The matmul
    // helper is invoked first so that its MinimalMatmulFusedOpSignaler is
    // populated with the matmul receiver cores' NOC coords + semaphore IDs
    // before the strided-all-gather kernels are emitted (the AG kernels read
    // back the populated signaler to know whom to signal once each tensor
    // slice is locally available).
    //
    // No workload-scoped resources are needed; GlobalSemaphores live on the
    // operation_attributes (caller-allocated; a different semaphore triggers
    // a recompile).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
