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
    bool use_block_interleaved = false;
    uint32_t cb_block_size_limit = 0;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

}  // namespace ttnn::prim
