// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_per_token_cast_back.hpp"

#include <tt_stl/assert.hpp>

#include "device/masked_per_token_cast_back_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back {

ttnn::Tensor masked_per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const std::optional<ttnn::Tensor>& input_scale,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t experts_per_chip,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& metadata) {
    // Exactly one scale source: either a plain fp32 (M, H/128) scale tensor, or the dispatch metadata
    // tensor whose row tail holds the per-token fp32 scales (read via scale_col_offset in the kernel).
    TT_FATAL(
        input_scale.has_value() != metadata.has_value(),
        "masked_per_token_cast_back: provide exactly one of `input_scale` or `metadata`");
    const bool scales_from_metadata = metadata.has_value();
    const ttnn::Tensor& scale_source = scales_from_metadata ? *metadata : *input_scale;
    return ttnn::prim::masked_per_token_cast_back(
        input_e4m3,
        scale_source,
        expert_region_offsets,
        expert_token_counts,
        global_expert_idx_table,
        experts_per_chip,
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        memory_config.value_or(input_e4m3.memory_config()),
        scales_from_metadata);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back
