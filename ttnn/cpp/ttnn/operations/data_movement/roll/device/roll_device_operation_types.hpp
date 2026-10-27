// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Single-dimension native sharded roll. The composite wrapper (roll.cpp) loops over dims,
// feeding each sharded result into the next, so the device op only handles one (shift, dim).
struct RollParams {
    uint32_t shift{};  // normalized to [0, dim_size)
    int32_t dim{};     // absolute (non-negative) dim index
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RollInputs {
    Tensor input;
};

}  // namespace ttnn::prim
