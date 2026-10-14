// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_lightning_select_kv.hpp"

#include "device/fused_lightning_select_kv_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> fused_lightning_select_kv(
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& head_weights,
    const Tensor& kv_cache,
    const Tensor& page_table_tensor,
    const Tensor& cur_pos_tensor,
    uint32_t k,
    const std::optional<Tensor>& valid_length_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& output_tensor,
    uint32_t output_row_offset,
    const std::optional<Tensor>& new_kv_row,
    const std::optional<Tensor>& new_kv_row_index,
    const std::optional<Tensor>& num_keys_tensor,
    uint32_t candidate_topk_blocks,
    uint32_t candidate_block_size,
    const std::optional<Tensor>& candidate_blocks_output,
    const std::optional<Tensor>& pool_count_output,
    uint32_t num_parts,
    const std::optional<Tensor>& part_index_tensor,
    const std::optional<Tensor>& row_table) {
    return ttnn::prim::fused_lightning_select_kv(
        query,
        key_cache,
        head_weights,
        kv_cache,
        page_table_tensor,
        cur_pos_tensor,
        k,
        valid_length_tensor,
        memory_config,
        compute_kernel_config,
        output_tensor,
        output_row_offset,
        new_kv_row,
        new_kv_row_index,
        num_keys_tensor,
        candidate_topk_blocks,
        candidate_block_size,
        candidate_blocks_output,
        pool_count_output,
        num_parts,
        part_index_tensor,
        row_table);
}

}  // namespace ttnn::experimental::deepseek
