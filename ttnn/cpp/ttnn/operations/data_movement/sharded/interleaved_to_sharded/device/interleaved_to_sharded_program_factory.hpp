// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "interleaved_to_sharded_op_types.hpp"

namespace ttnn::prim {

struct InterleavedToShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor);

    static bool uses_bank_segments(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);
};

}  // namespace ttnn::prim
