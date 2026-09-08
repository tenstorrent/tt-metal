// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "width_to_height_shard_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct WidthToHeightShardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const WidthToHeightShardParams& operation_attributes,
        const WidthToHeightShardInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
