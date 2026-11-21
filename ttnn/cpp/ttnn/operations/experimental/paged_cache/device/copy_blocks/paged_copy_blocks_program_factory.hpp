// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "paged_copy_blocks_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PagedCopyBlocksProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedCopyBlocksParams& attrs, const PagedCopyBlocksInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::experimental::prim
