// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "merge_select_kv.hpp"

#include "device/merge_select_kv_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor merge_select_kv(
    const Tensor& candidates,
    const Tensor& kv_cache,
    uint32_t k,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    uint32_t output_row_offset,
    uint32_t candidate_topk_blocks,
    uint32_t candidate_block_size,
    const std::optional<Tensor>& candidate_blocks_output,
    const std::optional<Tensor>& pool_count_output) {
    return ttnn::prim::merge_select_kv(
        candidates,
        kv_cache,
        k,
        memory_config,
        output_tensor,
        output_row_offset,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_blocks_output,
        pool_count_output);
}

}  // namespace ttnn::experimental::deepseek
