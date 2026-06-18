// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op factory: the immutable ProgramSpec plus the
// mutable ProgramRunArgs. Returned by a ProgramSpecFactoryConcept factory's
// create_program_spec method; the framework adapter stamps a Program out of
// this artifact onto each mesh coordinate range of the workload.
//
// A future MeshWorkloadSpecFactoryConcept will return a different (multi-program)
// artifact type for ops whose programs vary across the mesh.
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;

    // Op-owned device tensors: scratch / workspace / config tensors the factory
    // allocates itself (beyond the op's declared io). The adapter parks these in
    // the cache entry for the cached Program's lifetime — created once on a cache
    // miss, reused (never re-allocated) on a hit. A TensorArgument in run_params
    // may reference one of these (via its .mesh_tensor()) exactly as it would an
    // io tensor (matched by MeshTensor identity).
    //
    // These are full ttnn::Tensors (not bare MeshTensors) so the factory can hand
    // over tensors it has already *populated* with host data (config/index tables,
    // e.g. conv2d/pool/halo via move_config_tensor_to_device) and with arbitrary
    // (incl. sharded) placement — not just empty scratch. The ttnn::Tensor also
    // carries the device-buffer ownership that keeps the allocation alive while parked.
    //
    // FOOTGUN: build the run_params TensorArguments that reference these AGAINST
    // THE ELEMENTS OF THIS VECTOR (after populating it), never against a pre-move
    // local — the adapter resolves bindings by pointer identity (on .mesh_tensor()).
    // ttnn::Tensor is handle-backed, so its .mesh_tensor() address is stable across
    // vector moves; returning the artifact by value is safe.
    //
    // An op that allocates op-owned tensors is implicitly opted into the
    // "minimize cache-hit cost" hit path: on a cache hit the adapter does NOT
    // re-run the factory (which would re-allocate these); it reuses the parked
    // tensors and refreshes only the io tensor bindings.
    std::vector<tt::tt_metal::Tensor> op_owned_tensors;
};

}  // namespace ttnn::device_operation
