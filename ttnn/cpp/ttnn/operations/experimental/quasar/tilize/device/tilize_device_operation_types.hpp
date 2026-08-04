// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/kernel_types.hpp"

namespace ttnn::prim::qsr {

struct TilizeParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool use_multicore = false;
    // Host L1 heuristic: default multicore path CBs fit one full tile-row (tensor width in tiles).
    bool enough_space_width = false;
    const bool use_low_perf = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct TilizeInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_input_tensor;
};

}  // namespace ttnn::prim::qsr
