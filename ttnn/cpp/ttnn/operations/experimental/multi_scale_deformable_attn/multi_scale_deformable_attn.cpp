// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_scale_deformable_attn.hpp"

#include <optional>

#include "ttnn/operations/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor multi_scale_deformable_attn(
    const ttnn::Tensor& value,
    const ttnn::Tensor& grid,
    const ttnn::Tensor& attn,
    const std::optional<MemoryConfig>& memory_config,
    bool align_corners) {
    return ttnn::prim::multi_scale_deformable_attn(value, grid, attn, memory_config, align_corners);
}

}  // namespace ttnn::experimental
