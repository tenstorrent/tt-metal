// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmShardedHeightOnlyProgramFactory {
    // Every per-core argument is pinned by the hashed shapes and shard specs, and the two shard
    // base addresses ride borrowed-memory DFBs that the framework re-points from their
    // TensorArguments on every dispatch. Nothing else varies per dispatch, so this factory needs
    // no runtime-argument override. (Replaced get_dynamic_runtime_args, #48928.)
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
