// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back.hpp"

#include <tt_stl/assert.hpp>
#include "device/per_token_cast_back_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back {

ttnn::Tensor per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const std::optional<ttnn::Tensor>& input_scale,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool token_count_aware,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<ttnn::Tensor>& expert_token_counts,
    const std::optional<ttnn::Tensor>& global_expert_idx_table,
    uint32_t experts_per_chip,
    const std::optional<ttnn::Tensor>& metadata) {
    const auto dtype = output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16);
    const auto mem_config = memory_config.value_or(input_e4m3.memory_config());

    if (!token_count_aware) {
        TT_FATAL(
            input_scale.has_value(),
            "per_token_cast_back: input_scale is required (unless token_count_aware with metadata)");
        TT_FATAL(
            !metadata.has_value() && !expert_region_offsets.has_value() && !expert_token_counts.has_value() &&
                !global_expert_idx_table.has_value(),
            "per_token_cast_back: metadata / expert_* tensors are only valid when token_count_aware=true");
        return ttnn::prim::per_token_cast_back(input_e4m3, *input_scale, dtype, mem_config);
    }

    // Token-count-aware path. Exactly one scale source: either a plain fp32 (M, H/128) scale tensor, or
    // the dispatch metadata tensor whose row tail holds the per-token fp32 scales.
    TT_FATAL(
        input_scale.has_value() != metadata.has_value(),
        "per_token_cast_back: provide exactly one of `input_scale` or `metadata`");
    TT_FATAL(
        expert_region_offsets.has_value() && expert_token_counts.has_value() && global_expert_idx_table.has_value(),
        "per_token_cast_back: token_count_aware=true requires expert_region_offsets, expert_token_counts and "
        "global_expert_idx_table");
    const bool scales_from_metadata = metadata.has_value();
    const ttnn::Tensor& scale_source = scales_from_metadata ? *metadata : *input_scale;
    return ttnn::prim::per_token_cast_back(
        input_e4m3,
        scale_source,
        dtype,
        mem_config,
        /*token_count_aware=*/true,
        expert_region_offsets,
        expert_token_counts,
        global_expert_idx_table,
        experts_per_chip,
        scales_from_metadata);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back
