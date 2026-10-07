// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "rms_allgather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

// Kernel indices follow the push order in the per-coordinate descriptor builder, and the writer arg slot
// follows the writer runtime-arg layout; override_runtime_arguments() re-applies the semaphore address there.
namespace rms_allgather_dynamic {
inline constexpr uint32_t kWriterAllToAllKernelIdx = 0;
// Present only when the shard grid has workers outside the all-to-all set.
inline constexpr uint32_t kWriterNotAllToAllKernelIdx = 1;
// Writer layout: [0]=offset of post args, [1..4]=mcast rect, [5]=scaler, [6]=core id, [7]=out_ready_sem,
// [8]=out_ready_sem wait value, [9]=stats address (Buffer* binding).
inline constexpr uint32_t kWriterSemaphoreArg = 7;
inline constexpr uint32_t kWriterStatsAddrArg = 9;
// Offset of the gamma address (Buffer* binding) from the start of the post args.
inline constexpr uint32_t kWriterPostGammaAddrOffset = 2;
}  // namespace rms_allgather_dynamic

struct RMSAllGatherProgramFactory {
    // One ProgramDescriptor per coordinate: device index and fabric neighbors depend on the coordinate.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Tensor addresses are refreshed through Buffer* / CB bindings. This re-applies only the caller-supplied
    // GlobalSemaphore address, which RMSAllGatherDeviceOperation::compute_program_hash excludes from the key.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
