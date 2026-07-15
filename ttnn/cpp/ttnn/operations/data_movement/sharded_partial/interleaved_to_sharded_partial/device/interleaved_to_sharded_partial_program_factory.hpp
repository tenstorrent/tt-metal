// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "interleaved_to_sharded_partial_op_types.hpp"

namespace ttnn::prim {

struct InterleavedToShardedPartialProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const InterleavedToShardedPartialParams& params, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim
