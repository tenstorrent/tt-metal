// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

// DeepSeek V4 lightning-indexer selection for a prefill chunk: scores and top-k in one pass over the visible entries.
struct DeepseekV4PrefillIndexerSelectParams {
    uint32_t compress_ratio = 1;
    bool candidates = false;  // also select each query's candidate blocks (the two-level select's level one)
    DeviceComputeKernelConfig compute_kernel_config;
};

struct DeepseekV4PrefillIndexerSelectInputs {
    Tensor q;         // [1, H, T, D] bf16 TILE
    Tensor weights;   // [1, 1, T, H] bf16 TILE, gates already scaled
    Tensor idx_keys;  // [1, 1, E, D] bf16 ROW_MAJOR
    Tensor start;     // 1-element uint32/int32 ROW_MAJOR: the chunk's absolute start position
    // Pool mode: [1, 1, T, 2048] uint32 ROW_MAJOR candidate block ids per query (the candidate mode's output);
    // each query selects among its candidate blocks' entries only.
    std::optional<Tensor> candidates;
};

}  // namespace ttnn::prim
