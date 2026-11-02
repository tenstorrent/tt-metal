// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "dit_fused_distributed_rmsnorm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Kernel indices follow the push order in the per-coordinate descriptor builder; arg slots follow the
// kernels' runtime-arg layout. override_runtime_arguments() re-applies every address at these slots.
namespace dit_fused_distributed_rmsnorm_dynamic {
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterKernelIdx = 1;
inline constexpr uint32_t kComputeKernelIdx = 2;
// Forwarder f (all-gather path only) sits at kFirstForwarderKernelIdx + f.
inline constexpr uint32_t kFirstForwarderKernelIdx = 3;

// Reader common runtime args.
inline constexpr uint32_t kReaderInputAddrArg = 0;
inline constexpr uint32_t kReaderWeightAddrArg = 1;
inline constexpr uint32_t kReaderBiasAddrArg = 2;
inline constexpr uint32_t kReaderRopeCosAddrArg = 3;
inline constexpr uint32_t kReaderRopeSinAddrArg = 4;
inline constexpr uint32_t kReaderRecipAddrArg = 5;
// Present only with an affine_tile_row_map.
inline constexpr uint32_t kReaderTileRowMapAddrArg = 6;

// Writer common runtime args (drain-only and worker writer share slots 0 and 1).
inline constexpr uint32_t kWriterOutputAddrArg = 0;
inline constexpr uint32_t kWriterTransMatAddrArg = 1;
// Worker writer (all-gather path) only.
inline constexpr uint32_t kWriterStatsAddrArg = 2;

// Forwarder per-core runtime args.
inline constexpr uint32_t kForwarderStatsAddrArg = 0;
inline constexpr uint32_t kForwarderOutReadySemArg = 1;
}  // namespace dit_fused_distributed_rmsnorm_dynamic

struct DitFusedDistributedRmsnormMeshWorkloadFactory {
    // Local normalization (ring_size == 1 or per_head_norm) shares one ProgramDescriptor per range; the
    // all-gather path builds one per coordinate (device index and fabric neighbors depend on it).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DitFusedDistributedRmsnormParams& operation_attributes,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Re-applies the caller-supplied out_ready GlobalSemaphore address, which
    // DitFusedDistributedRmsnormDeviceOperation::compute_program_hash excludes from the key, and every tensor
    // address by role. Buffer* bindings alone are not enough: when two tensor arguments share a buffer (for
    // example weight and bias, or rope_cos and rope_sin), binding resolution yields no bindings, and the
    // workload path has no rebuild fallback.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const DitFusedDistributedRmsnormParams& operation_attributes,
        const DitFusedDistributedRmsnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
