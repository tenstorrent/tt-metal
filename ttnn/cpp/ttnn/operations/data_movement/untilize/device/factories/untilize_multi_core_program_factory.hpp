// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "../untilize_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& output);
};
}  // namespace ttnn::prim
