// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_fabric_bound_program_factory.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherMinimalMatmulAsyncProgramFactory {
    // Per-coord program build: the matmul kernels come first, then the strided all-gather pipeline.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Re-applies every tensor address by role (the all-gather output is also the matmul input) and every
    // caller-supplied semaphore address, which StridedAllGatherMinimalMatmulAsync::compute_program_hash leaves out.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
        const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
