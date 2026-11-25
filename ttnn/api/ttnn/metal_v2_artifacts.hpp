// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/memory/tensor/mesh_tensor.hpp>

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

    // Tensors the factory allocates for itself, for TensorArguments in run_params to reference. They are allocated
    // on every cache miss and held until the program cache is cleared (in practice, the whole model run), which
    // wastes scarce memory and fragments the allocator's address space. Every further cache miss of the op
    // allocates another copy, even an identical one, which lives just as long.
    // LEGACY ONLY: No new op should use this field. Much better to take the tensor as an input instead.
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
    // GlobalSemaphores the factory allocates, for GlobalSemaphoreArguments in run_params to reference (fill the
    // vector before taking references: growing it moves its elements). These have the same problem as
    // ProgramArtifacts::op_owned_tensors: allocated on every cache miss and held until the cache is cleared.
    // No better option exists yet: taking them from the caller (operation_attributes_t::global_semaphores) exposes
    // the op's internal synchronization in its interface, which TTNN deliberately moved away from.
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphores;

    // No op-owned tensors in MeshWorkloadArtifacts: see ProgramArtifacts::op_owned_tensors.

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
