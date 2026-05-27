// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 port: satisfies ProgramSpecFactoryConcept.
struct UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim
