// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/quasar/untilize/device/untilize_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct UntilizeMultiCoreBlockProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim::qsr
