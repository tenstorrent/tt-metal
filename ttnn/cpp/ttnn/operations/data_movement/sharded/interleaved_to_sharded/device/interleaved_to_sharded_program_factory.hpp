// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "interleaved_to_sharded_op_types.hpp"

namespace ttnn::prim {

struct InterleavedToShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
