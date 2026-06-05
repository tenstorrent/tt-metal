// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::device_operation {

// This file contains TTNN-specific data structures.
// These use Metalium's Metal 2.0 descriptor structs to create higher-level
// data structures, specific to the needs of TTNN ops.

/**
 * Declarative description of a single-device Program.
 * (Metal 2.0 equivalent to the legacy ProgramDescriptor.)
 *
 * ProgramSpec describes the immutable properties of a Program.
 * These are the Program properties that cannot be changed after Program construction.
 * The ProgramSpec is analogous to the signature and body of a "whole device function",
 * which Program construction compiles into an executable device object.
 *
 * ProgramRunArgs describes the mutable properties of a Program.
 * These are the Program properties that CAN be changed after Program construction.
 * The ProgramRunArgs are analogous to the arguments passed to the "function" above.
 * Fresh ProgramRunArgs are provided with each execution (enqueue) of the Program.
 *
 */
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;
};

/**
 * Declarative description of a mesh-scoped workload.
 *
 * A MeshWorkloadArtifacts pairs the per-coord ProgramArtifacts that make up
 * the workload with the workload-scoped resources the ProgramSpecs reference.
 * The framework realises it into a MeshWorkload on cache miss and keeps the
 * artifacts alive for the cached workload's lifetime so that:
 *   - GlobalSemaphores keep their device-side allocations valid, and
 *   - MeshTensors keep their device-side allocations valid.
 *     (Note: MeshTensor is an RAII mesh device memory object with unique
 *      ownership semantics.)
 *
 * Layout choices:
 *   - `programs` is range-keyed rather than coord-keyed so that ops with a
 *     single program replicated across the whole mesh emit just one entry.
 *   - Resources are flat vectors (not named slots) so factories with N
 *     semaphores or N buffers can grow without changing the schema.
 */
struct MeshWorkloadArtifacts {
    // Workload-scoped resources, allocated by the factory during workload
    // build and kept alive for the lifetime of the cached MeshWorkload.
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    std::vector<tt::tt_metal::MeshTensor> mesh_tensors;

    // Per-coord program descriptors.  Each entry covers a contiguous
    // MeshCoordinateRange and is materialised into one Program added to the
    // resulting MeshWorkload.
    struct PerCoordProgramArtifacts {
        tt::tt_metal::distributed::MeshCoordinateRange range;
        ProgramArtifacts artifacts;
    };
    std::vector<PerCoordProgramArtifacts> program_precursors;
};

}  // namespace ttnn::device_operation
