// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepseek_moe_reduce_scatter_device_operation_types.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterMeshWorkloadFactory {
    // Layout of WorkloadDescriptor::semaphores. Allocated once per cache miss and parked on the
    // descriptor, so their L1 addresses are stable for the cached workload.
    struct SemaphoreIndex {
        static constexpr size_t op = 0;
        static constexpr size_t pre_op_barrier = 1;
        static constexpr size_t count = 2;
    };

    // build_program_descriptor pushes reader, writer, then reduction.
    static constexpr uint32_t kReaderKernelIdx = 0;
    static constexpr uint32_t kWriterKernelIdx = 1;
    static constexpr uint32_t kReduceKernelIdx = 2;
    // Writer runtime args start with one address per intermediate slice, then the output.
    static constexpr uint32_t kWriterIntermediateArg0 = 0;
    static constexpr uint32_t kWriterOutputArg = 8;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const DeepseekMoEReduceScatterParams& operation_attributes,
        const DeepseekMoEReduceScatterInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
