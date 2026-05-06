// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatNewParams {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::CoreRangeSet> sub_core_grids;
};

struct ConcatNewInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim
