// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::prim {
struct AllBroadcastProgramFactory {
    // Declarative workload-scoped factory (Contract 2).
    //
    // create_workload_descriptor() runs ONCE per workload (cache miss):
    //   1. Allocates the two GlobalSemaphores used by writer runtime args
    //      (out-ready drain semaphore + cross-device init barrier semaphore)
    //      and parks them on WorkloadDescriptor::semaphores so they outlive
    //      the cached workload via the program cache.
    //   2. Runs the distributed Synchronize barrier so every device sees the
    //      semaphores before any program is dispatched.
    //   3. Loops `tensor_coords` and pushes a per-coord ProgramDescriptor
    //      into WorkloadDescriptor::programs.  Each program depends on the
    //      sender device coordinate (ring_index + forward/backward neighbor
    //      fabric nodes), so we build one descriptor per coord rather than
    //      sharing one across the mesh.
    //
    // Buffer base addresses are bound via emplace_runtime_args() so the
    // framework patches them through the fast cache-hit path; no manual
    // override_runtime_arguments() hook is required.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllBroadcastParams& operation_attributes,
        const Tensor& input,
        std::vector<Tensor>& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
