// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SDPAKSplitMergeParams {
    uint32_t k_split = 1;
    float scale = 1.0f;
    tt::tt_metal::MemoryConfig output_mem_config;
};

// Raw K-split partitions of ring joint SDPA (program_config.ring_k_split > 1): partial_output [B, k_split * NH, S, DV]
// unnormalized outputs, virtual head p * NH + h; partial_stats [B, k_split * NH, 2 S, 32] their running max (rows
// [0, S), column 0) and running sum (rows [S, 2 S), 32 per-column partials).
struct SDPAKSplitMergeInputs {
    ttnn::Tensor partial_output;
    ttnn::Tensor partial_stats;
};

}  // namespace ttnn::prim
