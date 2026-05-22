// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_strided_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncProgramFactory {
    // Contract-2: declarative WorkloadDescriptor.  Per coord, builds a
    // ProgramDescriptor by composing the ProgramDescriptor variants of the
    // strided reduce-scatter ring builder and the minimal_matmul helper.  The
    // RS builder is invoked first so that its
    // StridedReduceScatterFusedOpSignaler is populated with the RS reader
    // cores' NOC coords + semaphore IDs before the matmul kernels are emitted.
    // No workload-scoped resources are needed; GlobalSemaphores live on the
    // operation_attributes (caller-allocated; a different semaphore triggers a
    // recompile).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
