// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op factory: the immutable ProgramSpec plus the
// mutable ProgramRunArgs. Returned by a ProgramSpecFactoryConcept factory's
// create_program_spec method; the framework adapter stamps a Program out of
// this artifact onto each mesh coordinate range of the workload.
//
// A future MeshWorkloadSpecFactoryConcept will return a different (multi-program)
// artifact type for ops whose programs vary across the mesh.
struct ProgramArtifacts {
    tt::tt_metal::experimental::metal2_host_api::ProgramSpec spec;
    tt::tt_metal::experimental::metal2_host_api::ProgramRunArgs run_params;
};

}  // namespace ttnn::device_operation
