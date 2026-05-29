// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct MoeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MoeParams& operation_attributes, const MoeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
