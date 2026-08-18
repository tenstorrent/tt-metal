// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct BcastMultiCoreWProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
