// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "masked_bincount_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim::qsr {

struct MaskedBincountProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MaskedBincountParams& operation_attributes,
        const MaskedBincountInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::qsr
