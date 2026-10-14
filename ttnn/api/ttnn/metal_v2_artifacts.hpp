// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "ttnn/distributed/types.hpp"

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op-porting stepping-stone factory: the immutable
// ProgramSpec, the mutable ProgramRunArgs, and any op-owned tensors the factory
// allocates for itself. Returned by a ProgramSpecFactoryConcept or
// CustomProgramSpecFactoryConcept factory's
// create_program_artifacts method; the framework adapter maps that same spec
// onto tensor_coords via experimental::MakeMeshWorkloadFromSpecs.
//
// This artifact is SPMD-shaped: one program, replicated. Ops whose programs vary
// across the mesh return MeshWorkloadArtifacts instead.
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;

    // Op-owned device tensors the factory allocates for itself (scratch /
    // workspace). The adapter parks these in the program cache so their
    // device-memory allocation outlives the cache miss and stays at a stable
    // address across dispatches. Any TensorArgument in `run_params` may reference
    // one of these (by reference) in addition to the op's io tensors.
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
};

// Build product of a MeshWorkloadSpecFactoryConcept factory: the workload-scoped resources the
// factory allocates, plus the per-coordinate programs that reference them. Splits the same way
// tt::tt_metal::WorkloadDescriptor does, for the same reasons.
struct MeshWorkloadArtifacts {
    // Semaphores for cross-device sync belong here once Metal 2.0 has its own semaphore object.
    // No op-owned tensors: a MeshTensor spans the mesh, so op-allocated scratch stays SPMD-only.

    // Range-keyed: a program uniform over part of the mesh is one entry covering that range.
    struct PerCoordProgram {
        ttnn::MeshCoordinateRange range;
        tt::tt_metal::experimental::ProgramSpec spec;
        tt::tt_metal::experimental::ProgramRunArgs run_params;
    };
    std::vector<PerCoordProgram> programs;
};

}  // namespace ttnn::device_operation
