// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_planner.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::grouped_matmul {

namespace {

// plan() / bank_block_w()'s L1 rule: 2 x-blocks + 2 w-blocks + fp32 partials + the output tiles within kL1Budget.
bool block_fits(uint64_t mt, uint64_t bw, uint64_t pn, uint64_t a, uint64_t b, uint64_t o, uint64_t f32) {
    const uint64_t part = mt * pn * f32;
    return (2 * mt * bw * a) + (2 * bw * pn * b) + part + (mt * pn * o) <= kL1Budget;
}

// The widest block of the caller's block_w (alone, when given) or kBlockWidths that divides kt and fits; 0: none.
uint32_t pick_block_w(
    uint32_t mt, uint32_t kt, uint32_t pn, uint32_t a, uint32_t b, uint32_t o, uint32_t f32, uint32_t block_w) {
    const uint32_t given[] = {block_w};
    const std::span<const uint32_t> widths =
        block_w != 0 ? std::span<const uint32_t>(given) : std::span<const uint32_t>(kBlockWidths);
    for (const uint32_t bw : widths) {
        if (kt % bw == 0 && block_fits(mt, bw, pn, a, b, o, f32)) {
            return bw;
        }
    }
    return 0;
}

std::optional<InterleavedPlan> plan_uncached(const InterleavedPlanInputs& in) {
    if (in.x_rank != 4 || in.w_rank != 4 || in.x0 != 1 || in.w0 != 1) {
        return std::nullopt;
    }
    const uint32_t G = in.x1;
    const uint32_t M = in.x2;
    const uint32_t K = in.x3;
    if (in.w1 != G || in.w2 != K || M % tt::constants::TILE_HEIGHT != 0 || K % tt::constants::TILE_WIDTH != 0 ||
        in.w3 % tt::constants::TILE_WIDTH != 0) {
        return std::nullopt;
    }
    InterleavedPlan p;
    p.G = G;
    p.MT = M / tt::constants::TILE_HEIGHT;
    p.KT = K / tt::constants::TILE_WIDTH;
    p.NT = in.w3 / tt::constants::TILE_WIDTH;
    if (G > in.grid_y) {
        return std::nullopt;
    }
    p.GX = std::min(in.grid_x, p.NT);
    p.pns = column_split(p.NT, p.GX);
    p.BW = pick_block_w(
        p.MT, p.KT, p.pn_max(), in.a_tile_bytes, in.b_tile_bytes, in.o_tile_bytes, in.f32_tile_bytes, in.block_w);
    if (p.BW == 0) {
        return std::nullopt;
    }
    return p;
}

}  // namespace

std::vector<uint32_t> column_split(uint32_t nt, uint32_t gx) {
    // (the Python divmod(nt, 0) raises ZeroDivisionError)
    TT_FATAL(gx > 0, "grouped_matmul: column_split over {} grid columns", gx);
    const uint32_t base = nt / gx;
    const uint32_t rem = nt % gx;
    std::vector<uint32_t> pns(gx);
    for (uint32_t i = 0; i < gx; ++i) {
        pns[i] = base + (i < rem ? 1 : 0);
    }
    return pns;
}

const InterleavedPlan* solve_interleaved_plan(const InterleavedPlanInputs& in) {
    using Key = std::tuple<
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t>;
    // std::map nodes never move: a returned pointer stays valid for the process.
    static std::map<Key, std::optional<InterleavedPlan>> memo;
    static std::mutex mutex;
    const Key key{
        in.x_rank,
        in.x0,
        in.x1,
        in.x2,
        in.x3,
        in.w_rank,
        in.w0,
        in.w1,
        in.w2,
        in.w3,
        in.a_tile_bytes,
        in.b_tile_bytes,
        in.o_tile_bytes,
        in.f32_tile_bytes,
        in.grid_x,
        in.grid_y,
        in.block_w};
    const std::lock_guard<std::mutex> lock(mutex);
    auto it = memo.find(key);
    if (it == memo.end()) {
        it = memo.emplace(key, plan_uncached(in)).first;
    }
    return it->second.has_value() ? &*it->second : nullptr;
}

uint32_t solve_bank_block_w(
    uint32_t mt,
    uint32_t kt,
    uint32_t pn_max,
    uint32_t a_tile_bytes,
    uint32_t b_tile_bytes,
    uint32_t o_tile_bytes,
    uint32_t f32_tile_bytes,
    uint32_t block_w) {
    return pick_block_w(mt, kt, pn_max, a_tile_bytes, b_tile_bytes, o_tile_bytes, f32_tile_bytes, block_w);
}

}  // namespace ttnn::operations::grouped_matmul
