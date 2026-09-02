// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "fill_rm_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FillRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const FillRmParams& operation_attributes, const FillRmInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
