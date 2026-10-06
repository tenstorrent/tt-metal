// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"

#include <optional>

namespace tt::tt_metal {
class Program;
}

namespace ttnn::experimental::prim {

// ProgramDescriptor::kernels push order. Runtime-arg slots below are the positions
// emplace_runtime_args writes; buffer-address slots are Buffer* bindings, and the
// caller-owned GlobalSemaphore address is re-applied by override_runtime_arguments.
inline constexpr uint32_t kConcatReaderKernelIdx = 0;
inline constexpr uint32_t kTilizeWriterKernelIdx = 1;
inline constexpr uint32_t kTilizeComputeKernelIdx = 2;
inline constexpr uint32_t kWorkerReaderKernelIdx = 3;
inline constexpr uint32_t kWorkerWriterKernelIdx = 4;

inline constexpr uint32_t kWorkerReaderBufferArg = 0;
inline constexpr uint32_t kWorkerReaderSemaphoreArg = 1;
inline constexpr uint32_t kWorkerWriterBufferArg = 0;
inline constexpr uint32_t kWorkerWriterSemaphoreArg = 1;
inline constexpr uint32_t kConcatReaderBufferArg = 0;
inline constexpr uint32_t kConcatReaderInputBufferArg = 1;

struct AllGatherConcatMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherConcatParams& operation_attributes,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherConcatParams& operation_attributes,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
