// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::device_operation {

// Build product of a ProgramSpecFactoryConcept factory's create_program_spec: the immutable
// ProgramSpec (the program-cache key) plus the ProgramRunArgs that bind it to this dispatch's tensors.
struct ProgramSpecArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;
};

}  // namespace ttnn::device_operation
