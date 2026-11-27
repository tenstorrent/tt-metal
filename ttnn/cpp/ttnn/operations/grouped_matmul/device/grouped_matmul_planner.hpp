// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ttnn::operations::grouped_matmul {

// grouped_matmul.py L1_BUDGET: the bytes of CBs a core may take (the planner's own budget, not a device query).
constexpr uint32_t kL1Budget = 1024 * 1024;

// The block widths tried, widest first, when the caller gives no block_w.
constexpr uint32_t kBlockWidths[] = {32, 16, 8, 4, 2, 1};

// Inputs of grouped_matmul.py plan(): the shapes (rank-4 dims as given), tile bytes of x, w, out and float32
// (tt::tt_metal::tile_size(dtype)), the resolved grid and block_w (0: none).
struct InterleavedPlanInputs {
    uint32_t x_rank = 0;
    uint32_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;  // x shape [1, G, M, K]
    uint32_t w_rank = 0;
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;  // w shape [1, G, K, N]
    uint32_t a_tile_bytes = 0;
    uint32_t b_tile_bytes = 0;
    uint32_t o_tile_bytes = 0;
    uint32_t f32_tile_bytes = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t block_w = 0;
    bool operator==(const InterleavedPlanInputs&) const = default;
};

// plan()'s (G, MT, KT, NT, GX, pns, BW).
struct InterleavedPlan {
    uint32_t G = 0;
    uint32_t MT = 0;
    uint32_t KT = 0;
    uint32_t NT = 0;
    uint32_t GX = 0;
    std::vector<uint32_t> pns;  // column_split(NT, GX)
    uint32_t BW = 0;
    uint32_t pn_max() const { return pns.empty() ? 0 : pns.front(); }  // column_split puts the widest first
};

// column_split: nt column tiles over gx grid columns, as even as possible, the wider ones first.
std::vector<uint32_t> column_split(uint32_t nt, uint32_t gx);

// plan(): nullptr when the layout does not apply. Memoized on its inputs; the pointer stays valid for the process.
const InterleavedPlan* solve_interleaved_plan(const InterleavedPlanInputs& in);

// bank_block_w(): the bank path's K block, 0 when nothing fits L1. mt = M / 32, kt / pn_max from the layout.
uint32_t solve_bank_block_w(
    uint32_t mt,
    uint32_t kt,
    uint32_t pn_max,
    uint32_t a_tile_bytes,
    uint32_t b_tile_bytes,
    uint32_t o_tile_bytes,
    uint32_t f32_tile_bytes,
    uint32_t block_w);

}  // namespace ttnn::operations::grouped_matmul
