// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmShardedHeightOnlyProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
