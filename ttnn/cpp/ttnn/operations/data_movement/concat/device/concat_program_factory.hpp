// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct ConcatProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
