// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

// Contract-2 (descriptor) factory for the fused llama AllGather + Matmul op.
// Per coord the llama-specific multicast all-gather kernels and the llama
// 1D-gather_in0 matmul builder both append onto the same ProgramDescriptor;
// the MatmulFusedOpSignaler bridge between the two halves (LLAMA_ALL_GATHER
// variant) is identical to the legacy Program& factory.
struct LlamaAllGatherMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const LlamaAllGatherMatmulAsyncParams& operation_attributes,
        const LlamaAllGatherMatmulAsyncInputs& tensor_args,
        LlamaAllGatherMatmulAsyncResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
