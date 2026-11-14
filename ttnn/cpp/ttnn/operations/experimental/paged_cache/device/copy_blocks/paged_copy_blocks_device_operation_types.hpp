// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// A src block id that zeroes the dst block instead of copying.
inline constexpr uint32_t kPagedCopyBlocksZero = 0xFFFFFFFF;

struct PagedCopyBlocksParams {
    uint32_t pages_per_block;  // pages in one dim-0 slice (block) of every cache
};

struct PagedCopyBlocksInputs {
    std::vector<Tensor> caches;  // written in place; also the outputs
    Tensor src_dst;              // [num_pairs, 2] uint32 or int32, ROW_MAJOR: row i is (src block, dst block)
};

}  // namespace ttnn::experimental::prim
