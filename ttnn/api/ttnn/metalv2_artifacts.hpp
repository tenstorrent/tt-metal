// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::device_operation {

// Build product of a MetalV2 op factory's create_program_spec: the immutable ProgramSpec plus its
// run-args, split by enqueue cadence.
//   - invariant_run_args  — enqueue-invariant: set once on a cache miss and retained across hits.
//   - run_args — the per-enqueue set for the miss dispatch.
// The framework merges the two for the cache-miss SetProgramRunArgs. On a hit it re-applies only the
// per-enqueue set returned by create_per_enqueue_args (see operation_concepts.hpp).
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs invariant_run_args;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
};

}  // namespace ttnn::device_operation
