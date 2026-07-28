// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/tile.hpp>

namespace ttnn::prim {

struct TilizeParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool use_multicore = false;
    bool enough_space_width = false;
    bool enough_space_height = false;
    const bool use_low_perf = false;
    tt::tt_metal::Tile tile;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct TilizeInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_input_tensor;
};

}  // namespace ttnn::prim
