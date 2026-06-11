// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>
#include "ttnn/types.hpp"
#include "device/indexer_score_device_operation_types.hpp"

namespace ttnn::experimental::deepseek {

using IndexerScoreProgramConfig = ttnn::operations::experimental::deepseek::indexer::IndexerScoreProgramConfig;

// DeepSeek-V3.2 DSA lightning-indexer scorer:
//   score[b, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] -> score [B, 1, Sq, T] (row-major bf16).
// Causality from chunk_start_idx: key t visible to query s iff t <= chunk_start_idx + s.
// See models/demos/deepseek_v32/INDEXER_OP.md.
ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    bool is_causal = true,
    uint32_t chunk_start_idx = 0,
    const IndexerScoreProgramConfig& program_config = {});

}  // namespace ttnn::experimental::deepseek
