// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingParams {
    ttnn::Shape output_padded_shape{};
    tt::tt_metal::PadValue pad_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    bool use_multicore{};
    bool enough_space_width{};
    bool enough_space_height{};
    std::optional<CoreRangeSet> sub_core_grids;
};

}  // namespace ttnn::prim
