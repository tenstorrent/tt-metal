// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quasar/untilize/device/untilize_device_operation_types.hpp"
namespace ttnn::prim::qsr {

struct UntilizeSingleCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim::qsr
