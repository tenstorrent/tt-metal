// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

// Contract-2 (descriptor) factory for the fused Matmul + ReduceScatter op.
// Per coord the matmul (2D mcast) and ring reduce-scatter descriptor helpers
// both append onto the same ProgramDescriptor; the ReduceScatterFusedOpSignaler
// / MatmulFusedOpSignaler bridge between the two halves exactly as in the
// legacy Program& factory.
struct MatmulReduceScatterAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MatmulReduceScatterAsyncParams& args,
        const MatmulReduceScatterAsyncInputs& tensor_args,
        MatmulReduceScatterAsyncResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
