// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

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

    static constexpr auto attribute_names = std::forward_as_tuple(
        "output_padded_shape",
        "pad_value",
        "output_mem_config",
        "output_dtype",
        "use_multicore",
        "enough_space_width",
        "enough_space_height",
        "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(
            output_padded_shape,
            pad_value,
            output_mem_config,
            output_dtype,
            use_multicore,
            enough_space_width,
            enough_space_height,
            sub_core_grids);
    }
};

}  // namespace ttnn::prim
