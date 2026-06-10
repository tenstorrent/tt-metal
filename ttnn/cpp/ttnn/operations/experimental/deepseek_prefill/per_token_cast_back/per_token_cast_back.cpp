// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back.hpp"

#include "ttnn/operation.hpp"
#include "device/per_token_cast_back_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back {

ttnn::Tensor per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const ttnn::Tensor& input_scale,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& expert_token_counts,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<ttnn::Tensor>& metadata,
    std::optional<uint32_t> experts_per_chip,
    std::optional<uint32_t> dispatch_group_size) {
    // Masked mode is engaged iff the three routing tensors are supplied. They are all-or-nothing,
    // and the two scalar layout params must accompany them so the device op can compute the
    // per-device expert window (counter_offset).
    const bool masked = expert_token_counts.has_value() || expert_region_offsets.has_value() || metadata.has_value() ||
                        experts_per_chip.has_value() || dispatch_group_size.has_value();
    if (masked) {
        TT_FATAL(
            expert_token_counts.has_value() && expert_region_offsets.has_value() && metadata.has_value() &&
                experts_per_chip.has_value() && dispatch_group_size.has_value(),
            "per_token_cast_back: masked mode requires expert_token_counts, expert_region_offsets, metadata, "
            "experts_per_chip, and dispatch_group_size to all be provided together (or none of them)");
    }

    return ttnn::prim::per_token_cast_back(
        input_e4m3,
        input_scale,
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        memory_config.value_or(input_e4m3.memory_config()),
        expert_token_counts,
        expert_region_offsets,
        metadata,
        experts_per_chip.value_or(0),
        dispatch_group_size.value_or(0));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back
