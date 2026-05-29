// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

// Contract-2 (descriptor) factory for the fused AllGather + Minimal-Matmul op.
// All per-coord state (kernels, CBs, semaphores, fabric mux wiring) is built
// inside the per-coord ProgramDescriptor; the workload itself owns no extra
// resources beyond the caller-provided GlobalSemaphores referenced from
// operation_attributes.
struct AllGatherMinimalMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
