// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_strided_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncProgramFactory {
    // build_ring_strided_reduce_scatter_async_program_artifacts pushes mux, reader, writer, reduce first.
    static constexpr uint32_t kReaderKernelIdx = 1;
    static constexpr uint32_t kWriterKernelIdx = 2;
    static constexpr uint32_t kReduceKernelIdx = 3;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttnn::experimental::prim
