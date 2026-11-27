// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingParams {
    ttnn::Shape output_padded_shape{};
    ttnn::PadValue pad_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    bool use_multicore{};
    bool enough_space_width{};
    bool enough_space_height{};
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    // Opt-in: the multi-core interleaved program takes every count at run time and runs on the whole grid, so
    // one program serves every shape with the same last dims, at more host time per call.
    bool runtime_shapes = false;
};

}  // namespace ttnn::prim
