// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

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
};

}  // namespace ttnn::prim
