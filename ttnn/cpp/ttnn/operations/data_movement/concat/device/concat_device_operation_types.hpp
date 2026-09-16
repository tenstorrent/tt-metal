// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// ConcatParams::dim is already normalized, so "is this the height/width dim?" has to be asked
// rank-relative: at rank 3 the width dim is 2, at rank 5 it is 4. Asking with the absolute
// literals 2 and 3 is what #55032 was. Every sharded factory needs the same answer, so they all
// ask here rather than each rederiving it.
//
// Phrased as dim + N == rank so a rank-1 input cannot wrap the unsigned subtraction. Width is
// meaningful at rank 1 (dim 0 is the last dim); height is not, hence the guard.
constexpr bool is_width_concat(uint32_t rank, uint32_t dim) { return dim + 1 == rank; }
constexpr bool is_height_concat(uint32_t rank, uint32_t dim) { return rank >= 2 && dim + 2 == rank; }

}  // namespace ttnn::prim
