// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8.hpp"

#include "device/per_token_cast_to_fp8_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8 {

std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const ttnn::Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool round_scale_to_power_of_two,
    uint32_t scale_block_w,
    bool round_to_nearest_even,
    bool token_count_aware,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<ttnn::Tensor>& expert_token_counts,
    const std::optional<ttnn::Tensor>& global_expert_idx_table,
    uint32_t experts_per_chip,
    bool pack_scales_in_row,
    uint32_t packed_row_align) {
    return ttnn::prim::per_token_cast_to_fp8(
        input_tensor,
        memory_config.value_or(input_tensor.memory_config()),
        round_scale_to_power_of_two,
        scale_block_w,
        round_to_nearest_even,
        token_count_aware,
        expert_region_offsets,
        expert_token_counts,
        global_expert_idx_table,
        experts_per_chip,
        pack_scales_in_row,
        packed_row_align);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8
