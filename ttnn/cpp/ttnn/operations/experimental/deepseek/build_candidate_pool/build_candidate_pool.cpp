// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_candidate_pool.hpp"

#include "device/build_candidate_pool_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> build_candidate_pool(
    const Tensor& candidate_blocks,
    const Tensor& pool_count,
    const Tensor& key_cache,
    const Tensor& page_table_tensor,
    uint32_t candidate_block_size,
    const std::optional<Tensor>& pool_keys_output,
    const std::optional<Tensor>& pool_rows_output,
    const std::optional<Tensor>& page_table_row_tensor) {
    return ttnn::prim::build_candidate_pool(
        candidate_blocks,
        pool_count,
        key_cache,
        page_table_tensor,
        candidate_block_size,
        pool_keys_output,
        pool_rows_output,
        page_table_row_tensor);
}

}  // namespace ttnn::experimental::deepseek
