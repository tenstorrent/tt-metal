// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct SliceRmShardedProgramFactory {
    // Metal 2.0 (ProgramSpecFactoryConcept) factory for the height-sharded ROW_MAJOR
    // no-step slice path. Both dataflow buffers are *borrowed* (DataflowBufferSpec::borrowed_from
    // bound to the input/output TensorParameters) — the spec carries their sizing and the framework
    // re-binds their backing L1 address from the tensor argument on each execution. The reader's
    // per-core args are variable-length, so they are carried as per-node runtime varargs.
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
