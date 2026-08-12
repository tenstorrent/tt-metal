// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ShardedToInterleavedParams& operation_attributes,
        const ShardedToInterleavedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
