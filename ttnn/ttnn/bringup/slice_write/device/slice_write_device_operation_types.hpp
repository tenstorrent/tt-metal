// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>

#include <cstdint>
#include <vector>
#include <utility>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::bringup::prim::bringup {

struct SliceWriteParams {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
    // Bring-up option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18: at most 110 Tensix cores an op on a
    // Galaxy): the interleaved row-major work split runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1)
    // of the device's compute grid. Default (none) = the whole grid (the source op). Hashed when set.
    const std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

// The device's compute grid cut to max_grid (FIX.15)
inline tt::tt_metal::CoreCoord slice_write_grid(tt::tt_metal::CoreCoord g, const SliceWriteParams& p) {
    if (p.max_grid.has_value()) {
        g.x = std::max<std::size_t>(1, std::min<std::size_t>(g.x, p.max_grid->x));
        g.y = std::max<std::size_t>(1, std::min<std::size_t>(g.y, p.max_grid->y));
    }
    return g;
}

struct SliceWriteInputs {
    Tensor input;
    Tensor output;
};

using ReaderKernelArgs = std::vector<uint32_t>;
using WriterKernelArgs = std::vector<uint32_t>;
using KernelRuntimeArgs = std::pair<ReaderKernelArgs, WriterKernelArgs>;
using SliceWriteRuntimeArgs = std::vector<KernelRuntimeArgs>;

}  // namespace ttnn::experimental::bringup::prim::bringup
