// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct MoeAgRoutePlanParams {
    uint32_t experts_per_chip = 0;  // EPC: local slots 0 .. EPC - 1 in the local-slot map
    uint32_t num_rows = 0;          // rows of the flat expert space (token_index length)
};

// topk_indices: the gathered top-k expert ids [.., T, K] uint16 ROW_MAJOR (one page per token).
// local_slot_map: per device [.., 1, NG] uint32 ROW_MAJOR, global expert id -> local slot (< EPC) or 0xFFFFFFFF.
// preallocated_outputs: empty, or (counts, regions, token_index, y_slot) with the output specs.
struct MoeAgRoutePlanInputs {
    ttnn::Tensor topk_indices;
    ttnn::Tensor local_slot_map;
    std::vector<ttnn::Tensor> preallocated_outputs;
};

}  // namespace ttnn::prim
