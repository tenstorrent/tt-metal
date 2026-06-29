// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct ReshapeViewRMProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
