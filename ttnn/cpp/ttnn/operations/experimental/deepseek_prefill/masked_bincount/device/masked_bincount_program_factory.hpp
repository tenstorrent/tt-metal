// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "masked_bincount_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MaskedBincountProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  All CBs are local scratch
    // (per-shard input pages, output histogram, gather temp, mask).  Three
    // semaphores (init, done, gather) coordinate the tree-reduction.  Per-core
    // runtime args carry buffer addresses plus the per-level tree topology
    // (parent NOC coords and per-level child NOC coords).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MaskedBincountParams& operation_attributes,
        const MaskedBincountInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
