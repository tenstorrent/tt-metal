// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/normalization/all_gather_rms_norm/device/all_gather_rms_norm_device_operation_types.hpp"

namespace ttnn::prim {

// Declarative workload-scoped factory ("Contract 2", mirrors ccl/all_broadcast).
//
// create_workload_descriptor() runs ONCE per workload (cache miss):
//   1. Allocates the workload-scoped init-barrier GlobalSemaphore and parks it on
//      WorkloadDescriptor::semaphores so it outlives the cached workload.  The
//      user-supplied operation_attributes.semaphore is the all-gather out-ready
//      (drain) semaphore, referenced by absolute address from the writer args.
//   2. Runs the distributed Synchronize barrier so every device sees the semaphores
//      before any program is dispatched.
//   3. Loops tensor_coords and pushes one per-coord ProgramDescriptor into
//      WorkloadDescriptor::programs.  Each program depends on the sender device
//      coordinate (ring_index + forward/backward fabric neighbors), so we build one
//      descriptor per coord.
//
// Buffer base addresses are bound via KernelDescriptor::emplace_runtime_args() so the
// framework patches them through the fast cache-hit path; there is NO
// override_runtime_arguments() hook for declarative factories.
struct AllGatherRMSNormProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherRMSNormParams& operation_attributes,
        const AllGatherRMSNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
