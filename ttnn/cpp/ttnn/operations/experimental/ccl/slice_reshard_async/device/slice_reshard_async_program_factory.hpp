// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

// Kernel push order and runtime-arg slots shared by the descriptor builder and
// override_runtime_arguments(). Every worker core `{core_idx, 0}` (core_idx = link * kNumDirections + direction)
// gets its own reader kernel followed by its own writer kernel, so the reader of core_idx sits at
// kernel index core_idx * kKernelsPerCore + kReaderKernelOffset.
namespace slice_reshard_async_dynamic {
inline constexpr uint32_t kNumDirections = 2;
inline constexpr uint32_t kKernelsPerCore = 2;
inline constexpr uint32_t kReaderKernelOffset = 0;
inline constexpr uint32_t kWriterKernelOffset = 1;
inline constexpr uint32_t kReaderInputAddrArg = 0;
inline constexpr uint32_t kReaderOutReadySemArg = 9;
inline constexpr uint32_t kWriterInputAddrArg = 0;
inline constexpr uint32_t kWriterOutputAddrArg = 1;
inline constexpr uint32_t kWriterOutReadySemArg = 14;
inline constexpr uint32_t kWriterBarrierSemArg = 18;

inline constexpr uint32_t reader_kernel_idx(uint32_t core_idx) {
    return (core_idx * kKernelsPerCore) + kReaderKernelOffset;
}
inline constexpr uint32_t writer_kernel_idx(uint32_t core_idx) {
    return (core_idx * kKernelsPerCore) + kWriterKernelOffset;
}
}  // namespace slice_reshard_async_dynamic

struct SliceReshardAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SliceReshardAsyncParams& args,
        const Tensor& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied final/barrier GlobalSemaphore addresses on every cache hit.
    // Input/output addresses are Buffer* bindings patched by the framework before this runs.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceReshardAsyncParams& args,
        const Tensor& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
