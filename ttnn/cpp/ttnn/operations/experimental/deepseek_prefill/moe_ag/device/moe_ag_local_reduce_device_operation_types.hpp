// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct MoeAgLocalReduceParams {
    // 0: one pass over the column's T tokens; 1 / 2 (two mesh rows, fused send-back): the other row's tokens into
    // `other` / this row's tokens plus the peer's gathered phase-1 partial into `own`.
    uint32_t phase = 0;
    uint32_t chunk_size_per_chip = 0;  // S
    bool split = false;                // phase 0, two mesh rows: own [S, H] (this row) and other [S, H] outputs
    bool tiled = false;                // phase 0, not split: the [T, H] partials as bf16 tiles; phase 2: own as tiles
    uint32_t pairs_depth = 4;          // (y row, weight) pairs buffered per core
};

// y: the experts' outputs at their flat rows [.., rows, H] bf16 ROW_MAJOR; y_slot [1, T K] uint32 (route plan);
// weights: the gathered top-k weights [.., T, K] bf16 ROW_MAJOR; chip_info: per device [.., 1, 16] uint32 (word 0 the
// mesh row, 1 the other row's block start, 2 this row's); peer (phase 2): the gathered phase-1 partials [.., 2 S, H].
struct MoeAgLocalReduceInputs {
    ttnn::Tensor y;
    ttnn::Tensor y_slot;
    ttnn::Tensor weights;
    ttnn::Tensor chip_info;
    std::optional<ttnn::Tensor> peer;
    std::vector<ttnn::Tensor> preallocated_outputs;
};

}  // namespace ttnn::prim
