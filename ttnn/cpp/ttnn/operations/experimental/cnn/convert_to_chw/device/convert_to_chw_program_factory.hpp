// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "convert_to_chw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct ConvertToCHWProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConvertToCHWParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
