// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op factory's create_program_artifacts: the immutable ProgramSpec plus
// the run-args bundled with it (the enqueue-invariant set when the factory also provides
// create_per_enqueue_args, or the complete set when there is no split). The framework merges these with
// the per-enqueue set on a cache miss and retains them across hits.
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;

    // Device tensors the factory allocated itself (precomputed config/lookup tables, not in tensor_args).
    // The framework keeps them alive for the cached program's lifetime and appends them to the tensor
    // binding enumeration, so run_params.tensor_args may std::cref into them. Empty for most ops.
    std::vector<tt::tt_metal::Tensor> owned_tensors;
};

}  // namespace ttnn::device_operation
