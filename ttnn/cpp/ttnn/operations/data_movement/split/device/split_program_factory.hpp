// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/split/device/split_device_operation_types.hpp"

namespace ttnn::prim {

struct SplitProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SplitParams& operation_attributes,
        const SplitInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
