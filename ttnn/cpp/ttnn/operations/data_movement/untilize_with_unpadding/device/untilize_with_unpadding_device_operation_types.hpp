// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct UntilizeWithUnpaddingParams {
    ttnn::Shape output_tensor_end{};
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore = false;
    bool use_pack_untilize = false;
    bool fp32_dest_acc_en = false;
    bool enough_space_width = false;
    bool enough_space_height = false;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "output_tensor_end",
        "output_mem_config",
        "use_multicore",
        "use_pack_untilize",
        "fp32_dest_acc_en",
        "enough_space_width",
        "enough_space_height",
        "sub_core_grids");
    auto attribute_values() const {
        return std::forward_as_tuple(
            output_tensor_end,
            output_mem_config,
            use_multicore,
            use_pack_untilize,
            fp32_dest_acc_en,
            enough_space_width,
            enough_space_height,
            sub_core_grids);
    }
};

}  // namespace ttnn::prim
