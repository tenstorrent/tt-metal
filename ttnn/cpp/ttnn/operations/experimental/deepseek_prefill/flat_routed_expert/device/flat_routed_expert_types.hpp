// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "flat_routed_expert_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

using FlatRoutedExpertParams = FlatRoutedExpertConfig;

struct FlatRoutedExpertInputs {
    Tensor x;                                   // row-major bf16 dispatch buffer [rows, H], DRAM interleaved
    Tensor counts;                              // [1, NG] uint32 row major, DRAM interleaved: tokens per global expert
    Tensor regions;                             // [1, NG] uint32: each global expert's first row in x
    Tensor global_expert_ids;                   // [E] uint32 row major, DRAM: this device's local experts' global ids
    Tensor gate_up_weights;                     // per-reader bank regions (flat_routed_expert_weights layout)
    Tensor down_weights;                        // per-down-core bank regions
    std::optional<Tensor> reader_down_weights;  // per-reader-tail bank regions (plans with reader-down)
    Tensor done_words;  // persistent L1 words on the down coordinators (zero; the op leaves them zero)
    Tensor arena;       // per-launch L1 scratch on every role core (FlatRoutedExpertPlan::arena_tiles)
    Tensor words;       // per-launch L1 words on the x relays
    Tensor output;      // [rows, H] bfp8 TILE, DRAM interleaved (only active experts' rows written)
    // indexed mode: [1, rows] uint32 row-major DRAM; flat row r reads x row token_index[r] (x = e.g. all-gathered
    // tokens instead of a dispatch buffer)
    std::optional<Tensor> token_index;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
