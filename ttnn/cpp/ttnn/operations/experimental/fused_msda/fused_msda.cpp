// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_msda.hpp"

#include "ttnn/operations/experimental/fused_msda/device/fused_msda_device_operation.hpp"
#include "ttnn/operations/experimental/fused_msda/device/head_major_value_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor fused_msda(
    const ttnn::Tensor& value,
    const ttnn::Tensor& sampling_locations,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners,
    bool locations_in_grid_space,
    MSDAValueLayout value_layout,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::fused_msda(
        value,
        sampling_locations,
        attention_weights,
        spatial_shapes,
        align_corners,
        locations_in_grid_space,
        value_layout,
        memory_config);
}

ttnn::Tensor fused_msda_from_offsets(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    MSDAReferenceMode reference_mode,
    bool align_corners,
    MSDAValueLayout value_layout,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<uint32_t>>& num_valid_queries,
    const std::optional<ttnn::Tensor>& num_valid_queries_tensor) {
    return ttnn::prim::fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        reference_mode,
        align_corners,
        value_layout,
        memory_config,
        num_valid_queries,
        num_valid_queries_tensor);
}

bool can_use_fused_msda_from_offsets_l1_value(
    const ttnn::Tensor& value,
    const ttnn::Tensor& reference_points,
    const ttnn::Tensor& sampling_offsets,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    MSDAReferenceMode reference_mode,
    bool align_corners,
    bool device_valid_counts,
    MSDAValueLayout value_layout) {
    return ttnn::prim::can_use_fused_msda_from_offsets_l1_value(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        reference_mode,
        align_corners,
        device_valid_counts,
        value_layout);
}

ttnn::Tensor fused_msda_head_major_value(
    const ttnn::Tensor& value, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::fused_msda_head_major_value(value, num_heads, memory_config);
}

}  // namespace ttnn::experimental
