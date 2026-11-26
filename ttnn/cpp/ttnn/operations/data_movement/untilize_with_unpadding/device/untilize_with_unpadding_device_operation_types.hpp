// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct UntilizeWithUnpaddingParams {
    ttnn::Shape output_tensor_end{};
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore = false;
    bool fp32_dest_acc_en = false;
    bool enough_space_height = false;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids = std::nullopt;
    // Opt-in: the multi-core interleaved program takes every count at run time and runs on the whole grid, so
    // one program serves every shape with the same last dim, at more host time per call.
    bool runtime_shapes = false;
};

}  // namespace ttnn::prim
