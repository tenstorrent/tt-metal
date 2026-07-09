// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op-porting stepping-stone factory: the immutable
// ProgramSpec, the mutable ProgramRunArgs, and any op-owned tensors the factory
// allocates for itself. Returned by a MetalV2FactoryConcept factory's
// create_program_artifacts method; the framework adapter stamps a Program out of
// this artifact onto each mesh coordinate range of the workload.
//
// A future MeshWorkloadSpecFactoryConcept will return a different (multi-program)
// artifact type for ops whose programs vary across the mesh.
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

}  // namespace ttnn::device_operation
