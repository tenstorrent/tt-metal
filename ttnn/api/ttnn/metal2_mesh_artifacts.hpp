// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/distributed/types.hpp"

namespace ttnn::device_operation {

// Per-coordinate build product of a Metal 2.0 op factory:
// the immutable ProgramSpec plus the mutable ProgramRunParams.
//
// Returned (one per program in a workload) by an op factory's create_mesh_spec.
struct ProgramArtifacts {
    tt::tt_metal::experimental::metal2_host_api::ProgramSpec spec;
    tt::tt_metal::experimental::metal2_host_api::ProgramRunParams run_params;
};

// Workload-level build product of a Metal 2.0 op factory: one ProgramArtifacts
// per mesh coordinate range covered by the workload.
//
// Single-entry vector for the SPMD case (one program covering the full mesh
// range); multi-entry for mesh-distributed ops. Phase 2 will add
// workload-scoped state (GlobalSemaphore, op-owned MeshTensors, etc.) here as
// additional fields; the Phase 1 -> Phase 2 transition is purely additive.
struct MeshArtifacts {
    std::vector<std::pair<ttnn::MeshCoordinateRange, ProgramArtifacts>> programs;
};

}  // namespace ttnn::device_operation
