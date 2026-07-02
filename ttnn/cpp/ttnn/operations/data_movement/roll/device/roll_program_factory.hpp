// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "roll_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Native single-dim roll for ROW_MAJOR sharded tensors (HEIGHT / WIDTH / BLOCK).
// Implemented as a per-core gather of segment copies over the sharded buffers.
struct RollShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RollParams& operation_attributes, const RollInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
