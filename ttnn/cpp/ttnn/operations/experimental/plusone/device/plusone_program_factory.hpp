// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "plusone_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct PlusOneProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PlusoneParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
