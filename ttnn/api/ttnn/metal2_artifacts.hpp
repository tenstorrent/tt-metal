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
    // Op-owned device tensors (config / lookup / scratch) allocated by the
    // factory and kept alive for the lifetime of the cached Program(s).
    // MeshTensor is an RAII mesh device-memory object with unique ownership;
    // the framework moves these into the cache entry so their device
    // allocations outlive the (asynchronous) dispatch and stay valid across
    // cache hits.  Empty for the common case of an op that owns no resources.
    //
    // Only meaningful under the MinimizeCacheHitCost strategy: that path runs
    // the factory exactly once (on cache miss), so the tensors are allocated
    // once.  The default MaximizeCacheReuse path re-runs the factory on every
    // dispatch and would re-allocate them, so a non-empty op_owned_tensors is
    // rejected there (see dispatch_option2_spec_hash).
    //
    // GlobalSemaphores are intentionally NOT offered here: a semaphore is a
    // cross-program / cross-device coordination resource, which makes no sense
    // on a single Program.  Ops that need op-owned semaphores belong on the
    // multi-program workload concept (MeshWorkloadArtifacts).
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;

    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
};

/**
 * Declarative description of a mesh-scoped workload.
 *
 * A MeshWorkloadArtifacts pairs the per-coord programs that make up the
 * workload with the workload-scoped resources the ProgramSpecs reference.
 * The framework realises it into a MeshWorkload on cache miss and keeps the
 * resources alive for the cached workload's lifetime so that:
 *   - GlobalSemaphores keep their device-side allocations valid, and
 *   - MeshTensors keep their device-side allocations valid.
 *     (Note: MeshTensor is an RAII mesh device memory object with unique
 *      ownership semantics.)
 *
 * Layout choices:
 *   - `program_precursors` is range-keyed rather than coord-keyed so that ops
 *     with a single program replicated across the whole mesh emit just one
 *     entry.
 *   - Resources are flat vectors (not named slots) so factories with N
 *     semaphores or N tensors can grow without changing the schema.
 */
struct MeshWorkloadArtifacts {
    // Workload-scoped op-owned resources, allocated by the factory during
    // workload build and kept alive for the lifetime of the cached MeshWorkload.
    // These live at the workload level, NOT on the per-coord programs: a
    // workload's resources are shared across its programs, and the per-coord
    // unit is deliberately resource-free (storing spec + run_args directly
    // rather than a ProgramArtifacts, which would carry its own op_owned_tensors
    // and shadow these).
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphores;
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;

    // Per-coord programs.  Each entry covers a contiguous MeshCoordinateRange
    // and is materialised into one Program added to the resulting MeshWorkload.
    struct PerCoordProgram {
        tt::tt_metal::distributed::MeshCoordinateRange range;
        tt::tt_metal::experimental::ProgramSpec spec;
        tt::tt_metal::experimental::ProgramRunArgs run_args;
    };
    std::vector<PerCoordProgram> program_precursors;
};

}  // namespace ttnn::device_operation
