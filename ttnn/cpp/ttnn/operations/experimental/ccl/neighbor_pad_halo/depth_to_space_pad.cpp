// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "depth_to_space_pad.hpp"
#include "device/depth_to_space_pad_device_operation.hpp"

#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor depth_to_space_pad(
    const ttnn::Tensor& conv_out,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    bool drop_first,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(np_padding_h > 0, "depth_to_space_pad: np_padding_h must be > 0");

    MemoryConfig output_mem_config = memory_config.value_or(conv_out.memory_config());

    ttnn::experimental::prim::DepthToSpacePadParams params{
        .p1 = p1,
        .p2 = p2,
        .p3 = p3,
        .np_padding_h = np_padding_h,
        .np_padding_w = np_padding_w,
        .drop_first = drop_first ? 1u : 0u,
        .output_mem_config = output_mem_config,
    };

    return ttnn::prim::depth_to_space_pad(conv_out, params);
}

}  // namespace ttnn::experimental
