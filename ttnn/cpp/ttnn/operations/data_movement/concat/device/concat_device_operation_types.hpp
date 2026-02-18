// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatParams {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::CoreRangeSet> sub_core_grids;
};

struct ConcatInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim
