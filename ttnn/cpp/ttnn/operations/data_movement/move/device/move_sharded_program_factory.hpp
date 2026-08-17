// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "move_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct MoveShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
