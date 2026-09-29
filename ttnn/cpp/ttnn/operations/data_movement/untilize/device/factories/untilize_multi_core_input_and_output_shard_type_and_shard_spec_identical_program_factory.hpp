// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim
