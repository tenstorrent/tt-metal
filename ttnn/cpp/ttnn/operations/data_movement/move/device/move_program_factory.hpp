// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "move_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Program factory for MULTI_CORE and MULTI_CORE_OVERLAP strategies
struct MoveProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
