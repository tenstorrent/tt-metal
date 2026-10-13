// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_msda.hpp"

#include "ttnn/operations/experimental/fused_msda/device/fused_msda_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor fused_msda(
    const ttnn::Tensor& value,
    const ttnn::Tensor& sampling_locations,
    const ttnn::Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners,
    bool locations_in_grid_space,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::fused_msda(
        value,
        sampling_locations,
        attention_weights,
        spatial_shapes,
        align_corners,
        locations_in_grid_space,
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
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        spatial_shapes,
        reference_mode,
        align_corners,
        memory_config);
}

}  // namespace ttnn::experimental
