// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's layout planner: every candidate layout of the shape (matmul cores GX x GY, MT row tiles a core row,
// NCH row chunks, K step KB, the router's logits on the matmul cores or on router cores, P-tile or compact RS zones, the
// DRAM fallback), the ones whose L1 fits and whose per-core requirements hold (head_sizing.hpp), ranked by a cost model
// calibrated on measured heads. fit() is the head's whole host plan: a shape it accepts builds; a shape it refuses says which requirement
// failed, with the numbers and what would fit. Host only.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "head_sizing.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

// Host-only defaults (a shape planned without a device): a Galaxy chip's compute grid and free L1. A block plans on
// its device's grid and measured free L1 (fused_head_l1_free).
inline constexpr std::array<uint32_t, 2> kHeadGrid = {12, 10};  // head_plan.GRID
inline constexpr int64_t kL1Free = 1383 * 1024;                 // head_plan.L1_FREE
// L1 the block allocates after the head is planned (its own words: 128 B a bank in L1 with the persistent words in
// L1_SMALL, ~4.3 KB without) and the allocator's alignment
inline constexpr int64_t kL1Margin = 8 * 1024;                  // head_plan.L1_MARGIN
inline constexpr int64_t kUnlimited = int64_t(1) << 40;         // head_plan.UNLIMITED

// What the head computes, a chip (head_plan.HeadShape).
struct HeadShape {
    uint32_t rows = 0;  // a mesh row's tokens
    uint32_t H = 0;
    uint32_t I = 0;  // the shared expert's intermediate a TP chip
    uint32_t G = 0;  // TP
    uint32_t E = 0;  // router candidates (num_experts x fold)
    uint32_t K = 0;
    std::optional<uint32_t> latent;
    uint32_t fold = 1;
    uint32_t links = 2;
    bool ring = false;  // the TP axis wraps
    std::array<uint32_t, 2> grid = kHeadGrid;
    uint32_t R = 1;           // mesh rows
    bool sp_ring = false;     // the SP axis wraps
    uint32_t payload = 8256;  // the fabric's packet payload bytes
    uint32_t A = 16;          // the combine's accumulators
    WeightDtype w_dtype = WeightDtype::Bf8;

    int rs_dim() const { return latent ? -1 : -2; }
    uint32_t N() const { return R * G; }
    uint32_t S() const { return rows / G; }
    uint32_t EPC() const { return E / fold / N(); }
    bool operator==(const HeadShape&) const = default;
};

// A layout of the head; ordered field by field (the planner's tie order).
struct HeadLayout {
    uint32_t GY = 0;
    uint32_t MT = 0;
    uint32_t NCH = 0;
    uint32_t KB = 0;
    bool on_engine = false;
    uint32_t GX = 11;
    bool compact = false;  // compact RS arrival zones
    bool xdram = false;    // the DRAM fallback: x gathered into DRAM ahead of the program, reloaded a chunk at a time
    auto operator<=>(const HeadLayout&) const = default;
};

// The cost model's rates, us (head_plan.Cost).
struct HeadCost {
    double clk = 1.35e3;
    double tile = 20;
    double round_us = 0.9774;
    double link_gbs = 20.89;
    double dram_gbs = 244.5;
    double rtr_tile = 20;
    double chain_us = 83.11;
    double chunk_us = 48.04;
    double base_us = 26.21;
    double noc_gbs = 11.69;
    double chain_ek = 0.1852;
    double xmc_gbs = 43.98;
};

using Parts = std::vector<std::pair<std::string, int64_t>>;

// A sized layout and its predicted time (head_plan.Planned); t is +inf for a refused layout (note: why).
struct HeadPlanned {
    HeadLayout lay;
    int64_t arena = 0;  // bytes a core
    Regions sizes;
    double t = 0;
    Parts parts;
    std::string note;
};

// fit()'s answer (head_plan.HeadFit).
struct HeadFit {
    bool ok = false;
    std::vector<HeadPlanned> planned;  // best first
    std::string why;
    std::optional<HeadPlanned> free;  // the cost model's pick with unlimited L1
    std::string note;                 // what L1 forced, or the DRAM fallback's cost
};

// The arena cap a K step allows.
using ArenaMax = std::function<int64_t(uint32_t kb)>;
ArenaMax arena_max_of(int64_t l1_free, uint32_t payload, int64_t cb_floor = 0);
ArenaMax arena_max_fixed(int64_t bytes);

struct HeadFitOptions {
    std::optional<int64_t> l1_free;  // free L1 a core at the head's build (default kL1Free - kL1Margin)
    int64_t cb_floor = 0;  // other programs' static CBs under the arena (arena_cap_of: the MoE block's flat + dispatch)
    bool largest = true;   // on an L1 refusal, also the largest rows a mesh row that fit
    bool explain = true;   // the unconstrained pick and the L1 note
    bool fallback = true;  // MoEBlockConfig.head_dram_fallback
};

// head_plan.fit: the head's whole host plan.
HeadFit head_plan_fit(const HeadShape& shape, const HeadFitOptions& options = {});
HeadFit head_plan_fit(const HeadShape& shape, const ArenaMax& arena_max, const HeadFitOptions& options);

// The pieces fit() is built from (exposed for tests and the device op's build).
uint32_t engine_width(const HeadShape& shape);
std::vector<std::array<uint32_t, 3>> head_layouts(const HeadShape& shape, uint32_t mt_max = 2);  // (GY, MT, NCH)
std::vector<uint32_t> k_steps(const HeadShape& shape);
std::string shape_checks(const HeadShape& shape);
std::string describe_layout(const HeadLayout& lay);
// (sized engine or nullopt, arena bytes, region sizes, why not)
struct HeadSized {
    std::optional<SeEngineSizing> eng;
    int64_t arena = 0;
    Regions sizes;
    std::string why;
};
HeadSized head_size(const HeadShape& shape, const HeadLayout& lay, int64_t cap);
std::pair<double, Parts> head_cost(
    const HeadShape& shape, const HeadLayout& lay, const SeEngineSizing& eng, const HeadCost& c = {});
std::vector<HeadPlanned> head_plan_layouts(
    const HeadShape& shape, const ArenaMax& arena_max, bool keep_unfit = false, bool xdram = false);

// fused_head.router_rect: the columns of a rectangle of nrt router cores in `rows` free rows of `cols` cells, or 0.
uint32_t router_rect(uint32_t nrt, uint32_t cols, uint32_t rows);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
