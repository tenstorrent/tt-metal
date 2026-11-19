// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <hostdevcommon/kernel_structs.h>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "rms_allgather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

// Kernel indices follow the push order in the per-coordinate descriptor builder, and the writer arg slots follow
// the writer runtime-arg layout. override_runtime_arguments() re-applies the semaphore, stats and gamma addresses
// at these slots and repoints the tensor-backed circular buffers by CB index.
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

// Tensor-backed (globally allocated) circular buffers.
// Input; present only with a residual tensor.
inline constexpr uint32_t kOriginalInputCbIndex = tt::CBIndex::c_22;
// Residual, overwritten with input + residual; present only with a residual tensor.
inline constexpr uint32_t kUpdatedResidualCbIndex = tt::CBIndex::c_21;
// Residual when present, otherwise input.
inline constexpr uint32_t kIn0CbIndex = tt::CBIndex::c_12;
inline constexpr uint32_t kPreIn0CbIndex = tt::CBIndex::c_5;
// Output; tensor-backed only when the output shard spec equals the input shard spec (no write back).
inline constexpr uint32_t kOutputCbIndex = tt::CBIndex::c_10;
// Output; present only when the output is resharded (write back).
inline constexpr uint32_t kOutputReshardCbIndex = tt::CBIndex::c_16;
inline constexpr uint32_t kStatsCbIndex = tt::CBIndex::c_19;
}  // namespace rms_allgather_dynamic

struct RMSAllGatherProgramFactory {
    // One ProgramDescriptor per coordinate: device index and fabric neighbors depend on the coordinate.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied GlobalSemaphore address, which RMSAllGatherDeviceOperation::compute_program_hash
    // excludes from the key, and every tensor address by role: the writer stats/gamma slots and every tensor-backed
    // circular buffer. Buffer* / CB bindings alone are not enough: when two tensor arguments share a buffer (for
    // example a residual aliasing the input), binding resolution yields no bindings or maps them to the wrong tensor,
    // and the workload path has no rebuild fallback.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
