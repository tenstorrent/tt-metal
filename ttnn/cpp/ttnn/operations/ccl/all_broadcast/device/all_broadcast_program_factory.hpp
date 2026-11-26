// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::prim {
// A tiled interleaved input: the program copies its pages in per-link ranges that are runtime args, so one
// program serves every row count, and the hash leaves the rows out.
bool all_broadcast_program_is_row_free(const Tensor& input);

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
    // override_runtime_arguments() re-applies the buffer addresses and the
    // page ranges on a cache hit.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllBroadcastParams& operation_attributes,
        const Tensor& input,
        std::vector<Tensor>& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // On a cache hit: the buffer addresses, and each link's page range, which changes with the rows when the
    // program is row free.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllBroadcastParams& operation_attributes,
        const Tensor& input,
        std::vector<Tensor>& output_tensors,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttnn::prim
