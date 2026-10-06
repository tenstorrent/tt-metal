// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/enumerating_source.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include <fmt/format.h>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"
#include "ttnn/operations/matmul/device/config/factory_blocking_source.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

namespace {

struct Layout {
    Family family;
    Split split;
    BlockRules rules;

    bool operator==(const Layout&) const = default;
};

// The heuristic blocking, recording every layout it is asked to block
class RecordingBlocking final : public BlockingPolicy {
public:
    std::optional<Blocking> block(
        const MatmulDesc& p,
        const HardwareDesc& hw,
        Family family,
        const Split& split,
        const BlockRules& rules) const override {
        const Layout layout{family, split, rules};
        if (std::find(layouts_.begin(), layouts_.end(), layout) == layouts_.end()) {
            layouts_.push_back(layout);
        }
        return heuristic_.block(p, hw, family, split, rules);
    }

    const std::vector<Layout>& layouts() const { return layouts_; }

private:
    HeuristicBlocking heuristic_;
    mutable std::vector<Layout> layouts_;
};

// K depths to try: for each power of two up to Kt, the deepest allowed divisor of Kt at most that, and the
// layout's preferred depth
std::vector<uint32_t> k_depths(const MatmulDesc& p, const BlockRules& rules) {
    std::vector<uint32_t> allowed;
    for (uint32_t k : divisors_desc(p.Kt)) {
        if (rules.allows_k(k)) {
            allowed.push_back(k);
        }
    }
    std::set<uint32_t> depths;
    for (uint64_t cap = 1;; cap *= 2) {
        const auto deepest = std::find_if(allowed.begin(), allowed.end(), [&](uint32_t k) { return k <= cap; });
        if (deepest != allowed.end()) {
            depths.insert(*deepest);
        }
        if (cap >= p.Kt) {
            break;
        }
    }
    if (rules.k_preferred != 0 && rules.allows_k(rules.k_preferred) && p.Kt % rules.k_preferred == 0) {
        depths.insert(rules.k_preferred);
    }
    return {depths.begin(), depths.end()};
}

// The output blocks of a layout that fit L1 with `k`: of the blocks no other fitting block contains, the largest
// by area (then the squarer), then the tallest and the widest
std::vector<Blocking> output_blocks(const MatmulDesc& p, const HardwareDesc& hw, const Layout& layout, uint32_t k) {
    const Split& s = layout.split;
    const bool reuse = layout.family == Family::Reuse;
    const uint32_t rows = output_rows(p, s.fuse_batch);
    std::vector<Blocking> frontier;
    for (uint32_t h : reuse ? std::vector<uint32_t>{s.per_core_M} : divisors_desc(s.per_core_M)) {
        // in1-mcast with a single block row: rows % out_block_h == 0 or one output block per core
        if (layout.family == Family::Mcast1DIn1 && div_up(rows, s.per_core_M) == 1 && rows % h != 0 &&
            h != s.per_core_M) {
            continue;
        }
        // The widest block that fits at this height (the buffers only grow with the width)
        for (uint32_t w : reuse ? std::vector<uint32_t>{s.per_core_N} : divisors_desc(s.per_core_N)) {
            if (!layout.rules.allows_block_w(s.per_core_N, w)) {
                continue;
            }
            const Blocking b{s.per_core_M, s.per_core_N, k, h, w, 0, 0};
            if (circular_buffer_bytes(p, hw, layout.family, b, s.fuse_batch) <= hw.l1_cb_budget) {
                // A taller block already found at least this wide contains this one
                const bool contained = std::any_of(
                    frontier.begin(), frontier.end(), [&](const Blocking& f) { return f.out_block_w >= w; });
                if (!contained) {
                    frontier.push_back(b);
                }
                break;
            }
        }
    }
    auto area = [](const Blocking& b) { return static_cast<uint64_t>(b.out_block_h) * b.out_block_w; };
    auto skew = [](const Blocking& b) {
        return std::max(b.out_block_h, b.out_block_w) - std::min(b.out_block_h, b.out_block_w);
    };
    std::vector<Blocking> chosen;
    auto take = [&](auto better) {
        const Blocking* best = nullptr;
        for (const auto& b : frontier) {
            const bool taken = std::any_of(chosen.begin(), chosen.end(), [&](const Blocking& c) {
                return c.out_block_h == b.out_block_h && c.out_block_w == b.out_block_w;
            });
            if (!taken && (best == nullptr || better(b, *best))) {
                best = &b;
            }
        }
        if (best != nullptr && chosen.size() < EnumeratingSource::blocks_per_k) {
            chosen.push_back(*best);
        }
    };
    take([&](const Blocking& a, const Blocking& b) {
        return area(a) != area(b) ? area(a) > area(b) : skew(a) < skew(b);
    });
    take([](const Blocking& a, const Blocking& b) { return a.out_block_h > b.out_block_h; });
    take([](const Blocking& a, const Blocking& b) { return a.out_block_w > b.out_block_w; });
    return chosen;
}

// Subblocks to try in a block: the heuristic's, the widest single row, the tallest single column, and of the largest
// area the squarest, widest and tallest. Which shape is fastest depends on the operand formats, with no rule that
// holds.
std::vector<Blocking> subblock_variants(const MatmulDesc& p, Family family, const Blocking& heuristic) {
    const uint32_t limit = max_subblock_area(p, family);
    std::vector<std::pair<uint32_t, uint32_t>> shapes;
    for (uint32_t h : divisors_desc(heuristic.out_block_h)) {
        // Reuse's subblock rows must also divide each batch matrix's rows
        if (family == Family::Reuse && p.Mt % h != 0) {
            continue;
        }
        for (uint32_t w : divisors_desc(heuristic.out_block_w)) {
            if (h * w <= limit) {
                shapes.emplace_back(h, w);
            }
        }
    }
    if (shapes.empty()) {
        return {heuristic};
    }
    auto best = [&](auto better) {
        return *std::min_element(
            shapes.begin(), shapes.end(), [&](const auto& a, const auto& b) { return better(a, b); });
    };
    auto area = [](const auto& s) { return s.first * s.second; };
    auto skew = [](const auto& s) { return std::max(s.first, s.second) - std::min(s.first, s.second); };
    const std::pair<uint32_t, uint32_t> picks[] = {
        {heuristic.out_subblock_h, heuristic.out_subblock_w},
        best(
            [](const auto& a, const auto& b) { return (a.first == 1 ? a.second : 0) > (b.first == 1 ? b.second : 0); }),
        best(
            [](const auto& a, const auto& b) { return (a.second == 1 ? a.first : 0) > (b.second == 1 ? b.first : 0); }),
        best([&](const auto& a, const auto& b) { return area(a) != area(b) ? area(a) > area(b) : skew(a) < skew(b); }),
        best(
            [&](const auto& a, const auto& b) { return area(a) != area(b) ? area(a) > area(b) : a.second > b.second; }),
        best([&](const auto& a, const auto& b) { return area(a) != area(b) ? area(a) > area(b) : a.first > b.first; }),
    };
    std::vector<Blocking> result;
    for (const auto& [h, w] : picks) {
        const bool taken = std::any_of(result.begin(), result.end(), [&](const Blocking& b) {
            return b.out_subblock_h == h && b.out_subblock_w == w;
        });
        if (!taken) {
            Blocking b = heuristic;
            b.out_subblock_h = h;
            b.out_subblock_w = w;
            result.push_back(b);
        }
    }
    return result;
}

// The layout with the batch looped over instead of fused into M, split over the same cores
std::optional<Layout> looped_over_batch(const MatmulDesc& p, const HardwareDesc& hw, const Layout& layout) {
    if (layout.family == Family::Reuse || !layout.split.fuse_batch || p.batch_a <= 1) {
        return std::nullopt;
    }
    Layout looped = layout;
    looped.split.fuse_batch = false;
    switch (layout.family) {
        case Family::Mcast2D: looped.split.per_core_M = div_up(p.Mt, hw.grid.y); break;
        case Family::Mcast1DIn0: looped.split.per_core_M = p.Mt; break;
        case Family::Mcast1DIn1: looped.split.per_core_M = div_up(p.Mt, hw.grid.x * hw.grid.y); break;
        case Family::Reuse: break;
    }
    return looped;
}

}  // namespace

std::vector<Candidate> EnumeratingSource::propose(const MatmulDesc& p, const HardwareDesc& hw) const {
    auto recorder = std::make_shared<RecordingBlocking>();
    const FactoryBlockingSource heuristic(
        recorder, std::make_shared<HeuristicSubblock>(), std::make_shared<HeuristicFamily>());
    const std::vector<Candidate> chosen = heuristic.propose(p, hw);

    std::vector<Candidate> result;
    std::set<std::string> seen;
    auto add = [&](const Candidate& c) {
        const MatmulProgramConfig config = to_program_config(p, c);
        if (factory_limit_error(p, hw, config).empty() && seen.insert(fmt::format("{}", config)).second) {
            result.push_back(c);
        }
    };
    for (const auto& c : chosen) {
        add(c);
    }

    // Where each layout runs: a sharded layout on the heuristic candidate's grid (it is the only one), the
    // others on the whole grid
    std::vector<Layout> layouts;
    CoreCoord grid = hw.grid;
    std::optional<CoreRange> workers = pinned_workers(hw);
    bool transpose_mcast = false;
    if (sharded_layout(p)) {
        if (chosen.empty()) {
            return result;
        }
        const Candidate& c = chosen.front();
        grid = c.grid;
        workers = c.worker_cores;
        transpose_mcast = c.transpose_mcast;
        for (const auto& layout : recorder->layouts()) {
            if (layout.family == c.family && layout.split.per_core_M == c.blocking.per_core_M &&
                layout.split.per_core_N == c.blocking.per_core_N) {
                layouts.push_back(layout);
            }
        }
    } else {
        for (const auto& layout : recorder->layouts()) {
            layouts.push_back(layout);
            if (auto looped = looped_over_batch(p, hw, layout);
                looped && std::find(layouts.begin(), layouts.end(), *looped) == layouts.end()) {
                layouts.push_back(*looped);
            }
        }
    }

    HardwareDesc on_grid = hw;
    on_grid.grid = grid;
    const HeuristicSubblock subblock;
    for (const auto& layout : layouts) {
        for (uint32_t k : k_depths(p, layout.rules)) {
            for (const auto& b : output_blocks(p, hw, layout, k)) {
                for (const auto& blocked :
                     subblock_variants(p, layout.family, subblock.subblock(p, layout.family, b))) {
                    add(
                        {layout.family,
                         blocked,
                         cores_used(p, on_grid, layout.family, blocked, layout.split.fuse_batch),
                         grid,
                         workers,
                         transpose_mcast,
                         layout.split.fuse_batch});
                }
            }
        }
    }
    return result;
}

}  // namespace ttnn::operations::matmul::auto_config
