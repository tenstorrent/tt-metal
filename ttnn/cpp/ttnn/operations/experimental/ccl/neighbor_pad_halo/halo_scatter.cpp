// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "halo_scatter.hpp"
#include "device/halo_scatter_device_operation.hpp"

#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor halo_scatter(
    const ttnn::Tensor& compact_buffer,
    const ttnn::Tensor& interior_src,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(np_padding_h > 0, "halo_scatter: np_padding_h must be > 0");

    MemoryConfig output_mem_config = memory_config.value_or(interior_src.memory_config());

    ttnn::experimental::prim::NpHaloScatterParams params{
        .np_padding_h = np_padding_h,
        .np_padding_w = np_padding_w,
        .output_mem_config = output_mem_config,
    };

    return ttnn::prim::halo_scatter(compact_buffer, interior_src, params);
}

}  // namespace ttnn::experimental
