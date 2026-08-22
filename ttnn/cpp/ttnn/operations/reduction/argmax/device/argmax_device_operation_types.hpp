// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::prim {

struct ArgmaxParams {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    tt::tt_metal::MemoryConfig output_mem_config;
    // Opt-in: single-core RVV (Zve32f) scan on the pack RISC for TILE-layout
    // last-dim argmax (Blackhole only). Also enables the optional max-value
    // output (see ArgmaxInputs::optional_maxval_tensor).
    bool use_rvv{false};
};

struct ArgmaxInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
    // Optional preallocated BFLOAT16 ROW_MAJOR tensor (same logical shape as
    // the index output). When provided on the RVV path, the winning max
    // VALUES are written alongside the indices — greedy-sampling callers
    // then don't need a separate ttnn.max over the logits.
    std::optional<Tensor> optional_maxval_tensor;
};

}  // namespace ttnn::prim
