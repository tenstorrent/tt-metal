// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::paged_pool_gather {

struct operation_attributes_t {
    uint32_t pool_size{4};   // rows per pool (divides 16)
    uint32_t num_pools{0};   // N: pools per user in pool_ids
    uint32_t out_blocks{0};  // ceil(N * pool_size / 32): compact blocks per user
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    Tensor cache;       // [max_blocks, 1, 32, W] bf16 TILE
    Tensor page_table;  // [B_table, blocks_per_seq] int32 ROW_MAJOR
    Tensor pool_ids;    // [B, 1, 1, N] uint32 ROW_MAJOR
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttnn::operations::experimental::paged_pool_gather
