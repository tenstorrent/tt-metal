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

// Kernel indices follow the push order in the per-coordinate descriptor builder (reader, writer, forwarders,
// compute), which TT_FATALs on a cache miss if a pushed index differs from these; forwarder arg slots follow the
// forwarder runtime-arg layout. override_runtime_arguments() patches these.
// The reader and writer common-arg slot order is owned by the factory's address tables.
namespace dit_fused_distributed_rmsnorm_dynamic {
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterKernelIdx = 1;
// Forwarder f (all-gather path only) sits at kFirstForwarderKernelIdx + f.
inline constexpr uint32_t kFirstForwarderKernelIdx = 2;

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
