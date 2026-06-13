// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::device_operation {

// Build product of the Metal 2.0 "stepping stone" factory: everything needed to stamp and run one
// Program, produced in a single create_everything() call. Returned by an
// IntermediateStepMetalV2FactoryConcept factory; the framework adapter stamps a Program out of `spec`
// onto each mesh coordinate range of the workload, applies `run_params`, and takes ownership of any
// `op_owned_tensors`.
//
// This artifact is deliberately the *whole* picture, not a split. The stepping-stone concept does not
// participate in the fast-path cache-hit machinery: every dispatch rebuilds the full run args (and any
// op-owned tensors) from scratch. That is slow but trivially correct — there is no enqueue-invariant /
// per-enqueue partitioning for a porter to get wrong. The real (fast) concepts split this product back
// apart; see operation_concepts.hpp.
//
// MeshTensor is move-only (sole owner of its device memory), so this artifact is move-only.
struct MetalV2IntermediateStepArtifact {
    // Metal 2.0 structures: the immutable blueprint plus the full (merged) run args.
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;

    // Op-owned resources: device tensors the op allocates for itself — scratch / workspace, or a lookup
    // table for math the Tensix engine handles poorly (e.g. a trig table). The framework takes ownership
    // and keeps them alive in the program-cache entry for as long as the stamped Program references them.
    // Because the stepping-stone concept reallocates these on every dispatch, the adapter enqueues
    // *blocking* so the device finishes with the previous set before it is replaced. Empty for ops that
    // express all tensor needs through tensor_args / tensor_return_value.
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
};

}  // namespace ttnn::device_operation
