// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

// Metal 2.0 factory — degenerate ProgramSpecFactoryConcept.
//
// sharded_to_interleaved reads an L1-sharded input and writes an interleaved output. The sharded input
// CB is a DFB borrowed from the input tensor; the interleaved output is reached through a real
// TensorAccessor (ta::output) on the writer. Everything the program needs (the per-core work split,
// shape scalars) is a pure function of the request, so create_program_artifacts returns the spec PLUS
// the complete run-args; the cache-hit path only refreshes the tensor bindings (UpdateTensorArgs). No
// extract_immutable_info, no create_per_enqueue_args, no custom hash.
struct ShardedToInterleavedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
