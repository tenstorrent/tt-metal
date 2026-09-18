// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include "fast_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FastReduceNCProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const FastReduceNCParams& operation_attributes,
        const FastReduceNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
