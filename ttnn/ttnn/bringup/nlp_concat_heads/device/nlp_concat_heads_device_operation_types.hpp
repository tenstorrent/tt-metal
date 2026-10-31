// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::experimental::bringup::prim::bringup {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    // Bring-up option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18: at most 110 Tensix cores an op on a
    // Galaxy): the interleaved work split runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1) of the
    // device's compute grid. Default (none) = the whole grid (the source op). Part of the program hash.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

}  // namespace ttnn::experimental::bringup::prim::bringup
