// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "ttnn/distributed/types.hpp"

namespace ttnn::device_operation {

// Build product of a Metal 2.0 factory: the immutable
// ProgramSpec, the mutable ProgramRunArgs, and any op-owned tensors the factory
// allocates for itself. Returned by a ProgramSpecFactoryConcept or
// CustomProgramSpecFactoryConcept factory's
// create_program_artifacts method; the framework adapter maps that same spec
// onto tensor_coords via the MeshWorkload constructor.
//
// This artifact is SPMD-shaped: one program, replicated. Ops whose programs vary
// across the mesh return MeshWorkloadArtifacts instead.
struct ProgramArtifacts {
    tt::tt_metal::ProgramSpec spec;
    tt::tt_metal::ProgramRunArgs run_params;

    // Op-owned device tensors the factory allocates for itself (scratch /
    // workspace). The adapter parks these in the program cache so their
    // device-memory allocation outlives the cache miss and stays at a stable
    // address across dispatches. Any TensorArgument in `run_params` may reference
    // one of these (by reference) in addition to the op's io tensors.
    //
    // CAUTION: LEGACY ONLY. NO NEW OP MAY USE THIS FIELD, and no port may reach for it to make a port easier.
    //          It exists only to carry the few legacy ops that already allocate their own tensors across to
    //          Metal 2.0 unchanged. Every program cache entry holds its own copy, which stays allocated until the
    //          entry is dropped, long after the op's last use of it; the caller can neither see nor free that
    //          memory, and it fragments the device's address space. A new op takes such a tensor as an input,
    //          so that its caller owns the allocation.
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
};

// One device's values of named kernel run args (runtime or common, scalar or array) that differ from the
// range's base run args. Only named kernel run args: no tensor args, DFB overrides, varargs or advanced args.
struct DeviceRunArgs {
    ttnn::MeshCoordinate device;
    tt::tt_metal::ProgramRunArgs values;
};

// Build product of a MeshWorkloadSpecFactoryConcept factory: the workload-scoped resources the
// factory allocates, plus the per-coordinate programs that reference them. Splits the same way
// tt::tt_metal::WorkloadDescriptor does, for the same reasons.
struct MeshWorkloadArtifacts {
    // GlobalSemaphores the factory allocates for its programs (cross-device sync, say). The adapter parks them in
    // the program cache, so they stay allocated, at the same addresses, for every call that hits the cache entry.
    // A GlobalSemaphoreArgument in a program's run_params references one of these, or one of the caller's
    // GlobalSemaphores in operation_attributes_t::global_semaphores (which every cache hit rebinds). Fill this
    // vector before taking references into it: growing it moves its elements.
    //
    // CAUTION: Like tt::tt_metal::WorkloadDescriptor::semaphores, which it mirrors for ported ops, a parked
    //          GlobalSemaphore stays allocated until its program cache entry is dropped, long after the op's last
    //          use of it. A new op takes its GlobalSemaphores from its caller instead.
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphores;

    // No op-owned tensors, deliberately: see the CAUTION on ProgramArtifacts::op_owned_tensors.

    // Range-keyed: a program uniform over part of the mesh is one entry covering that range.
    struct PerCoordProgram {
        ttnn::MeshCoordinateRange range;
        tt::tt_metal::ProgramSpec spec;
        tt::tt_metal::ProgramRunArgs run_params;
        // Per-device values of named kernel run args (factories with `uses_per_device_run_args`, see
        // MeshWorkloadSpecFactoryAdapter): the range runs one program, and each listed device gets these values
        // on top of run_params. Every arg listed here must also be listed in run_params.
        std::vector<DeviceRunArgs> per_device_run_params;
    };
    std::vector<PerCoordProgram> programs;
};

// Cache-hit product of a MeshWorkloadSpecFactoryConcept factory with `uses_per_device_run_args`: the run args of
// every device of the range, and the per-device values of the devices that differ (see PerCoordProgram).
struct RangeRunArgs {
    tt::tt_metal::ProgramRunArgs base;
    std::vector<DeviceRunArgs> per_device;
};

}  // namespace ttnn::device_operation
