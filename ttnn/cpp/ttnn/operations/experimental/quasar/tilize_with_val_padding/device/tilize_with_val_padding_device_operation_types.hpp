// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

struct TilizeWithValPaddingParams {
    ttnn::Shape output_padded_shape{};
    tt::tt_metal::PadValue pad_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    tt::tt_metal::Tile tile{};
    bool use_multicore{};
    bool enough_space_width{};
    bool enough_space_height{};
    std::optional<CoreRangeSet> sub_core_grids;
};

}  // namespace ttnn::prim::qsr
