// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct TypecastRowMajorChunkedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
