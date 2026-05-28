// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

// Metal 2.0 port: this factory now satisfies ProgramSpecFactoryConcept.
// See METAL2_PORT_PLAN.md / METAL2_PORT_REPORT.md alongside this directory
// for context, design decisions, and friction notes.
struct ShardedToInterleavedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
