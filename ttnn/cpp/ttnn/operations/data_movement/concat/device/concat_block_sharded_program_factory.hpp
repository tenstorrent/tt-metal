// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct ConcatBlockShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
