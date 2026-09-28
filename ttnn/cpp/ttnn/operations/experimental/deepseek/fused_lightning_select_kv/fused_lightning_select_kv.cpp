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
    const std::optional<Tensor>& new_kv_row_index) {
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
        new_kv_row_index);
}

}  // namespace ttnn::experimental::deepseek
