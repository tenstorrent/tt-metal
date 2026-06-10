// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op factory's create_program_artifacts: the immutable ProgramSpec plus
// the run-args bundled with it (the enqueue-invariant set when the factory also provides
// create_per_enqueue_args, or the complete set in the degenerate case). The framework merges these with
// the per-enqueue set on a cache miss and retains them across hits.
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;
};

}  // namespace ttnn::device_operation
