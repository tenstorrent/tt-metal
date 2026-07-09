// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct ShardedToInterleavedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
