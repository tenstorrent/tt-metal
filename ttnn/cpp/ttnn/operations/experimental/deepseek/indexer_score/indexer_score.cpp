// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score.hpp"
#include "device/indexer_score_device_operation.hpp"

namespace ttnn::experimental::deepseek {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    bool is_causal,
    uint32_t chunk_start_idx,
    const IndexerScoreProgramConfig& program_config) {
    return ttnn::prim::indexer_score(q, k, weights, is_causal, chunk_start_idx, program_config);
}

}  // namespace ttnn::experimental::deepseek
