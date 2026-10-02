// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/factory_blocking_source.hpp"

#include <algorithm>
#include <utility>

#include "ttnn/operations/matmul/device/config/auto_config_common.hpp"
#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"

namespace ttnn::operations::matmul::auto_config {

using namespace detail;

// ---- Blocking ----

namespace {

// Largest in0_block_w (see HeuristicBlocking::Params). With a single K block the mcast factories single-buffer
// the inputs, so reading the next block can't overlap math on the current one: keep two blocks when K allows it.
// The reuse factory always double-buffers.
// Precision is the compute config's call: with packer L1 accumulation off (or when the factory doesn't use it),
// partial sums go through the output format between K blocks, and the blocking doesn't try to avoid that.
uint32_t max_in0_block_w(
    const HeuristicBlocking::Params& params,
    const MatmulDesc& p,
    Family family,
    uint32_t out_block_h,
    uint32_t out_block_w) {
    const uint32_t Kt = p.Kt;
    const uint32_t two_blocks = (family != Family::Reuse && Kt >= 2) ? Kt / 2 : Kt;
    // Tiles per K step of the operand(s) each core reads itself rather than receiving by multicast
    uint32_t self_read = 0;
    switch (family) {
        case Family::Mcast2D: self_read = 0; break;
        case Family::Mcast1DIn0: self_read = out_block_w; break;
        case Family::Mcast1DIn1: self_read = out_block_h; break;
        case Family::Reuse: self_read = out_block_h + out_block_w; break;
    }
    uint32_t depth = params.tuned.max_in0_block_w;
    // Interleaved 2D where K blocks are costly. Every K block ends with a fixed cost: a handshake, and a pack of the
    // whole output block's partials, which without packer L1 accumulation the next block reloads. It dominates
    // with a block-float input (a K block moves few input bytes for it) or, for a block at least 2 tiles on each
    // side, with L1 accumulation off (the pack and reload grow with the block; a one-tile-tall or -wide block pays
    // little and deeper K would only lengthen its fill). Then K may be split into as few as Tuned::max_costly_k_blocks
    // blocks, and the block search trades output block size against that depth. With accumulation on and 16-bit
    // inputs the cost hides, and the depth cap stays max_in0_block_w.
    const bool block_float = is_block_float(p.in0_format) || is_block_float(p.in1_format);
    const bool reloads_partials = !p.packer_l1_acc && std::min(out_block_h, out_block_w) >= 2;
    if (family == Family::Mcast2D && (block_float || reloads_partials) && !sharded_layout(p) &&
        params.tuned.max_costly_k_blocks != 0) {
        depth = std::max(depth, div_up(Kt, params.tuned.max_costly_k_blocks));
    }
    const uint32_t self_read_limit =
        self_read == 0
            ? depth
            : std::max(params.limits.min_in0_block_w, params.tuned.max_self_read_tiles_per_k_step / self_read);
    return std::min({depth, two_blocks, self_read_limit});
}

bool k_allowed(const BlockRules& rules, uint32_t k) {
    return (rules.k_fixed == 0 || k == rules.k_fixed) && (rules.k_divides == 0 || rules.k_divides % k == 0);
}

// Validation also admits out_block_h == 1 with a narrower out_block_w, but the 1D in0-mcast factory then
// writes a sharded output wrongly (#58046), so a sharded output's blocks always span per_core_N.
bool block_allowed(const BlockRules& rules, uint32_t per_core_N, uint32_t out_block_w) {
    return !rules.sharded_out || out_block_w == per_core_N;
}

// 2D mcast (issue #57884 heuristic 1): largest in0_block_w * out_block_h * out_block_w that fits L1 (among
// blocks at the layout's preferred in0_block_w, if any fit, then among blocks that fit with in0_block_w at least
// Limits::min_in0_block_w when Tuned::k_depth_over_block_size is on: every K block ends with a pack of the whole
// output block, which on some architectures doesn't hide behind data movement); ties go to the larger output block,
// then the squarer one (each loaded A and B tile is reused across the block's width and height, so a square block
// reuses the most for its area).
std::optional<Blocking> block_2d(
    const HeuristicBlocking::Params& params,
    const MatmulDesc& p,
    const HardwareDesc& hw,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const BlockRules& rules) {
    const auto k_options = divisors_desc(p.Kt);
    std::optional<Blocking> best;
    uint64_t best_product = 0;
    uint64_t best_area = 0;
    const uint32_t min_k = params.tuned.k_depth_over_block_size ? std::min(params.limits.min_in0_block_w, p.Kt) : 1;
    auto deep_enough = [&](uint32_t k) { return rules.k_fixed != 0 || k >= min_k; };
    for (uint32_t h : divisors_desc(per_core_M)) {
        for (uint32_t w : divisors_desc(per_core_N)) {
            const uint64_t area = static_cast<uint64_t>(h) * w;
            // K depth limit of this block size (larger blocks may go deeper); it only shrinks as w does
            const uint32_t k_max =
                rules.k_fixed != 0 ? rules.k_fixed : max_in0_block_w(params, p, Family::Mcast2D, h, w);
            if (best && !rules.prefers_other(best->in0_block_w) && deep_enough(best->in0_block_w) &&
                area * std::max(k_max, rules.k_preferred) < best_product) {
                break;  // narrower blocks for this h can't win
            }
            if (!block_allowed(rules, per_core_N, w)) {
                continue;
            }
            for (uint32_t k : k_options) {
                if ((k > k_max && !rules.prefers(k)) || !k_allowed(rules, k)) {
                    continue;
                }
                Blocking b{per_core_M, per_core_N, k, h, w, 0, 0};
                if (circular_buffer_bytes(p, hw, Family::Mcast2D, b, fuse_batch) > hw.l1_cb_budget) {
                    continue;
                }
                const uint64_t product = area * k;
                const uint32_t skew = h > w ? h - w : w - h;
                const uint32_t best_skew =
                    best ? (best->out_block_h > best->out_block_w ? best->out_block_h - best->out_block_w
                                                                  : best->out_block_w - best->out_block_h)
                         : 0;
                const bool preference = best && rules.prefers(k) != rules.prefers(best->in0_block_w);
                const bool depth = best && deep_enough(k) != deep_enough(best->in0_block_w);
                if (preference ? rules.prefers(k)
                    : depth    ? deep_enough(k)
                               : (product > best_product || (product == best_product && area > best_area) ||
                               (product == best_product && area == best_area && skew < best_skew))) {
                    best = b;
                    best_product = product;
                    best_area = area;
                }
                break;  // largest fitting k for this output block
            }
        }
    }
    return best;
}

// The widest divisor of `n` that is at most `limit`.
uint32_t widest_divisor_within(uint32_t n, uint32_t limit) {
    for (uint32_t d = std::min(n, limit); d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// 1D mcast (issue #57884 heuristic 2): keep the full per-core extent along the multicast dimension, shrink
// the other one only if needed; in0_block_w is the largest that fits, within the K depth rule (or the layout's
// preferred one, as in 2D). Two refinements:
//  - in 1D in0-mcast the core that reads B also writes the output, and a single output block wider than a
//    subblock row queues all of its writes after the last read: the block is split to the widest subblock
//    width (when that is at least 2 tiles; a sharded output keeps full-width blocks);
//  - if keeping the full multicast extent only fits with single-tile K steps, both dimensions are searched
//    as in 2D (largest in0_block_w * area; ties avoid 1-tile dimensions, then prefer the larger, squarer block).
std::optional<Blocking> block_1d(
    const HeuristicBlocking::Params& params,
    const MatmulDesc& p,
    const HardwareDesc& hw,
    Family family,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const BlockRules& rules) {
    const bool is_tall = family == Family::Mcast1DIn1;
    const uint32_t fixed_full = is_tall ? per_core_N : per_core_M;
    const uint32_t cheap_full = is_tall ? per_core_M : per_core_N;
    const uint32_t M_rows = output_rows(p, fuse_batch);
    const uint32_t max_area = max_subblock_area(p, family);
    const uint32_t split_w = widest_divisor_within(per_core_N, max_area);
    const bool split = !is_tall && rules.k_fixed == 0 && !rules.sharded_out && per_core_N > max_area && split_w > 1;

    // The largest fitting in0_block_w for this output block, if any
    auto fit = [&](uint32_t out_block_h, uint32_t out_block_w) -> std::optional<Blocking> {
        // in1-mcast with a single block row: Mt % out_block_h == 0 or one output block per core
        if (is_tall && div_up(M_rows, per_core_M) == 1 && M_rows % out_block_h != 0 && per_core_M != out_block_h) {
            return std::nullopt;
        }
        if (!block_allowed(rules, per_core_N, out_block_w) || (split && out_block_w > split_w)) {
            return std::nullopt;
        }
        const uint32_t k_limit =
            rules.k_fixed != 0 ? rules.k_fixed : max_in0_block_w(params, p, family, out_block_h, out_block_w);
        for (uint32_t k : divisors_desc(p.Kt)) {
            if ((k > k_limit && !rules.prefers(k)) || !k_allowed(rules, k)) {
                continue;
            }
            Blocking b{per_core_M, per_core_N, k, out_block_h, out_block_w, 0, 0};
            if (circular_buffer_bytes(p, hw, family, b, fuse_batch) <= hw.l1_cb_budget) {
                return b;
            }
        }
        return std::nullopt;
    };
    auto area_of = [](const Blocking& b) { return static_cast<uint64_t>(b.out_block_h) * b.out_block_w; };

    std::optional<Blocking> best;
    uint64_t best_product = 0;
    for (uint32_t cheap : divisors_desc(cheap_full)) {
        const auto b = fit(is_tall ? cheap : fixed_full, is_tall ? fixed_full : cheap);
        if (b) {
            const uint64_t product = area_of(*b) * b->in0_block_w;
            const bool preference = best && rules.prefers(b->in0_block_w) != rules.prefers(best->in0_block_w);
            if (preference ? rules.prefers(b->in0_block_w)
                           : (product > best_product || (product == best_product && area_of(*b) > area_of(*best)))) {
                best = b;
                best_product = product;
            }
        }
        if (best && (best->in0_block_w > 1 || best_product == static_cast<uint64_t>(per_core_M) * per_core_N)) {
            break;  // full block already fits at k >= 1; don't split further
        }
    }
    if (!best || best->in0_block_w > 1 || p.Kt == 1 || rules.k_fixed != 0 || rules.sharded_out) {
        return best;
    }

    // Single-tile K steps: shrink the multicast dimension too
    auto unit_dims = [&](const Blocking& b) {
        return (b.in0_block_w == 1) + (b.out_block_h == 1 && per_core_M > 1) + (b.out_block_w == 1 && per_core_N > 1);
    };
    auto skew = [](const Blocking& b) {
        return b.out_block_h > b.out_block_w ? b.out_block_h - b.out_block_w : b.out_block_w - b.out_block_h;
    };
    std::optional<Blocking> alt;
    for (uint32_t fixed : divisors_desc(fixed_full)) {
        for (uint32_t cheap : divisors_desc(cheap_full)) {
            const auto b = fit(is_tall ? cheap : fixed, is_tall ? fixed : cheap);
            if (!b) {
                continue;
            }
            const uint64_t product = area_of(*b) * b->in0_block_w;
            const uint64_t alt_product = alt ? area_of(*alt) * alt->in0_block_w : 0;
            bool better = !alt || product > alt_product;
            if (alt && product == alt_product) {
                if (unit_dims(*b) != unit_dims(*alt)) {
                    better = unit_dims(*b) < unit_dims(*alt);
                } else if (area_of(*b) != area_of(*alt)) {
                    better = area_of(*b) > area_of(*alt);
                } else {
                    better = skew(*b) < skew(*alt);
                }
            }
            if (better) {
                alt = b;
            }
        }
    }
    return alt && alt->in0_block_w > 1 ? alt : best;
}

// Reuse: per_core_N = Nt, and the deepest in0_block_w within the K depth rule that fits
std::optional<Blocking> block_reuse(
    const HeuristicBlocking::Params& params,
    const MatmulDesc& p,
    const HardwareDesc& hw,
    const Split& split,
    const BlockRules& rules) {
    const uint32_t k_limit = rules.k_fixed != 0
                                 ? rules.k_fixed
                                 : max_in0_block_w(params, p, Family::Reuse, split.per_core_M, split.per_core_N);
    for (uint32_t k : divisors_desc(p.Kt)) {
        if ((k > k_limit && !rules.prefers(k)) || !k_allowed(rules, k)) {
            continue;
        }
        Blocking b{split.per_core_M, split.per_core_N, k, split.per_core_M, split.per_core_N, 0, 0};
        if (circular_buffer_bytes(p, hw, Family::Reuse, b, true) <= hw.l1_cb_budget) {
            return b;
        }
    }
    return std::nullopt;
}

}  // namespace

HeuristicBlocking::Params HeuristicBlocking::Params::for_arch(tt::ARCH arch) {
    Params params;
    if (arch == tt::ARCH::BLACKHOLE) {
        params.tuned.max_self_read_tiles_per_k_step = 12;
        params.tuned.k_depth_over_block_size = true;
    }
    return params;
}

std::optional<Blocking> HeuristicBlocking::block(
    const MatmulDesc& p, const HardwareDesc& hw, Family family, const Split& split, const BlockRules& rules) const {
    const Params params = params_.value_or(Params::for_arch(hw.arch));
    switch (family) {
        case Family::Mcast2D: {
            auto b = block_2d(params, p, hw, split.per_core_M, split.per_core_N, split.fuse_batch, rules);
            return b;
        }
        case Family::Mcast1DIn0:
        case Family::Mcast1DIn1:
            return block_1d(params, p, hw, family, split.per_core_M, split.per_core_N, split.fuse_batch, rules);
        case Family::Reuse: return block_reuse(params, p, hw, split, rules);
    }
    return std::nullopt;
}

// ---- Subblocks ----

namespace {

// Largest-area subblock dividing (block_h, block_w). Among equal areas, with `prefer_two_wide` one at least
// two tiles on each side, then the wider one. `h_divides` restricts the height further (Reuse with batched
// inputs needs out_subblock_h | Mt); with `full_row_w` set, a subblock taller than one tile must span that
// width (sharded outputs: out_subblock_w == per_core_N or h == 1).
std::pair<uint32_t, uint32_t> choose_subblock(
    uint32_t block_h,
    uint32_t block_w,
    uint32_t max_area,
    bool prefer_two_wide,
    uint32_t h_divides = 0,
    uint32_t full_row_w = 0) {
    std::pair<uint32_t, uint32_t> best{1, 1};
    for (uint32_t h = 1; h <= std::min(block_h, max_area); ++h) {
        if (block_h % h != 0 || (h_divides != 0 && h_divides % h != 0)) {
            continue;
        }
        for (uint32_t w = std::min(block_w, max_area / h); w >= 1; --w) {
            if (block_w % w != 0 || (full_row_w != 0 && h > 1 && w != full_row_w)) {
                continue;
            }
            const auto area = h * w;
            const auto best_area = best.first * best.second;
            const bool two_wide = std::min(h, w) >= 2;
            const bool best_two_wide = std::min(best.first, best.second) >= 2;
            const bool tie_better = prefer_two_wide && two_wide != best_two_wide ? two_wide : w > best.second;
            if (area > best_area || (area == best_area && tie_better)) {
                best = {h, w};
            }
            break;  // widest w for this h found
        }
    }
    return best;
}

}  // namespace

// Subblocks two tiles or more on each side, unless B's tiles are smaller than A's. Per K step, an h x w
// subblock (h <= w) unpacks h tiles of A and h * w of B, so going from 1 x 8 to 2 x 4 costs one more A tile
// per 8 outputs, and avoids the single-row path, whose per-tile overhead shows as up to 10% on large
// matmuls with bf16 B. Only when A is the heavier operand (e.g. bf16 A, block-float B) does the extra A
// unpack cost more than it saves (1 x 8 then wins by ~5% on the Wormhole sweeps).
Blocking HeuristicSubblock::subblock(const MatmulDesc& p, Family family, Blocking b) const {
    const bool reuse = family == Family::Reuse;
    const bool prefer_two_wide = tt::tile_size(p.in1_format) >= tt::tile_size(p.in0_format);
    // Reuse with batched A and B requires out_subblock_h | Mt
    const auto [h, w] = choose_subblock(
        reuse ? b.per_core_M : b.out_block_h,
        reuse ? b.per_core_N : b.out_block_w,
        max_subblock_area(p, family),
        prefer_two_wide,
        reuse ? p.Mt : 0,
        p.out.sharded() ? b.per_core_N : 0);
    b.out_subblock_h = h;
    b.out_subblock_w = w;
    return b;
}

// ---- Family ----

namespace {

// Input tiles per K tile each core reads: its rows of A plus its columns of B
uint32_t per_core_input_tiles(const Blocking& b) { return b.per_core_M + b.per_core_N; }

// Input tiles read from memory in total. The mcast layouts read A once per output column block and B once
// per output row block (each shared by multicast); Reuse reads A once and B once per M slice of a batch.
uint64_t total_input_tiles(const MatmulDesc& p, Family family, const Blocking& b) {
    const uint64_t a_tiles = static_cast<uint64_t>(p.batch_a) * p.Mt * p.Kt;
    const uint64_t b_tiles = static_cast<uint64_t>(p.batch_b) * p.Kt * p.Nt;
    if (family == Family::Reuse) {
        return a_tiles + b_tiles * div_up(p.Mt, b.per_core_M);
    }
    return a_tiles * (b.per_core_N / b.out_block_w) + b_tiles * (b.per_core_M / b.out_block_h);
}

uint32_t cores_used(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch) {
    if (family == Family::Reuse) {
        const uint32_t blocks = p.batch_a * p.Mt / b.per_core_M;
        return std::min(blocks, static_cast<uint32_t>(hw.grid.x * hw.grid.y));
    }
    return div_up(output_rows(p, fuse_batch), b.per_core_M) * div_up(p.Nt, b.per_core_N);
}

// Among the multicast layouts: 2D unless a 1D layout keeps clearly more cores busy (in0-mcast first when both
// do), or 1D in0-mcast keeps as many cores busy with less input per core.
const Candidate* choose_mcast(
    double one_d_core_advantage, const Candidate* two_d, const Candidate* in0, const Candidate* in1) {
    if (!two_d) {
        return in0 ? in0 : in1;
    }
    const double threshold = one_d_core_advantage * two_d->cores;
    if (in0 && in0->cores >= threshold && (!in1 || in0->cores >= in1->cores)) {
        return in0;
    }
    if (in1 && in1->cores >= threshold) {
        return in1;
    }
    if (in0 && in0->cores >= two_d->cores &&
        per_core_input_tiles(in0->blocking) < per_core_input_tiles(two_d->blocking)) {
        return in0;  // a taller, squarer per-core block reads less input
    }
    return two_d;
}

std::optional<Candidate> choose_family(
    double one_d_core_advantage, const MatmulDesc& p, std::span<const Candidate> all) {
    auto find = [&](Family family) -> const Candidate* {
        for (const auto& c : all) {
            if (c.family == family) {
                return &c;
            }
        }
        return nullptr;
    };
    const Candidate* mcast =
        choose_mcast(one_d_core_advantage, find(Family::Mcast2D), find(Family::Mcast1DIn0), find(Family::Mcast1DIn1));
    // Batched B: Reuse, whose cores work on their blocks independently, unless the multicast layout (which
    // loops over the batch across the whole grid) keeps clearly more cores busy, or Reuse would read clearly
    // more input (it re-reads a batch's B for every M slice it splits the batch matrix into).
    if (const auto* reuse = find(Family::Reuse)) {
        if (mcast && (mcast->cores >= one_d_core_advantage * reuse->cores ||
                      total_input_tiles(p, Family::Reuse, reuse->blocking) >=
                          one_d_core_advantage * total_input_tiles(p, mcast->family, mcast->blocking))) {
            return *mcast;
        }
        return *reuse;
    }
    if (mcast) {
        return *mcast;
    }
    return std::nullopt;
}

// A 2D layout whose per-core blocks are one tile tall or wide multicasts single tiles along that axis, so the
// reuse the rules above count on isn't there: the roofline estimate picks the family instead (ties to the
// earlier family: 2D, 1D in0, 1D in1, Reuse).
bool one_tile_2d(const Candidate& c) {
    return c.family == Family::Mcast2D && std::min(c.blocking.per_core_M, c.blocking.per_core_N) == 1;
}

}  // namespace

HeuristicFamily::Params HeuristicFamily::Params::for_arch(tt::ARCH /*arch*/) { return {}; }

std::optional<Candidate> HeuristicFamily::choose(
    const MatmulDesc& p, const HardwareDesc& hw, std::span<const Candidate> all) const {
    const Params params = params_.value_or(Params::for_arch(hw.arch));
    auto chosen = choose_family(params.tuned.one_d_core_advantage, p, all);
    if (!chosen || !one_tile_2d(*chosen)) {
        return chosen;
    }
    auto key = [&](const Candidate& c) {
        return std::make_pair(roofline(p, hw, c.family, c.blocking, c.fuse_batch).cycles(), static_cast<int>(c.family));
    };
    return *std::min_element(
        all.begin(), all.end(), [&](const Candidate& x, const Candidate& y) { return key(x) < key(y); });
}

// ---- Source ----

FactoryBlockingSource::FactoryBlockingSource() :
    FactoryBlockingSource(
        std::make_shared<HeuristicBlocking>(),
        std::make_shared<HeuristicSubblock>(),
        std::make_shared<HeuristicFamily>()) {}

FactoryBlockingSource::FactoryBlockingSource(
    std::shared_ptr<const BlockingPolicy> blocking,
    std::shared_ptr<const SubblockPolicy> subblock,
    std::shared_ptr<const FamilyPolicy> family) :
    blocking_(std::move(blocking)), subblock_(std::move(subblock)), family_(std::move(family)) {}

std::vector<Candidate> FactoryBlockingSource::propose(const MatmulDesc& p, const HardwareDesc& hw) const {
    const auto all = candidates(p, hw);
    // A sharded layout fixes the family: its candidate is the only one
    auto chosen = sharded_layout(p) ? (all.empty() ? std::nullopt : std::optional<Candidate>(all.front()))
                                    : family_->choose(p, hw, all);
    if (!chosen) {
        return {};
    }
    // With a batched A fused into M, a layout that splits only M (1D in1, or 2D with blocks one tile wide) gives
    // each core one tall block, whose output is written after its last K step. Looping over the batch instead gives
    // each core one shorter block per batch, whose output writes overlap the next one's compute, provided one batch
    // still keeps as many cores busy. Decided after the family, which the estimate compares with the batch fused.
    const bool splits_only_m =
        chosen->family == Family::Mcast1DIn1 || (chosen->family == Family::Mcast2D && chosen->blocking.per_core_N == 1);
    if (!sharded_layout(p) && splits_only_m && chosen->fuse_batch && p.batch_a > 1) {
        const uint32_t rows = chosen->family == Family::Mcast2D ? hw.grid.y : hw.grid.x * hw.grid.y;
        if (auto looped =
                blocking_->block(p, hw, chosen->family, {div_up(p.Mt, rows), chosen->blocking.per_core_N, false}, {})) {
            // Only when one batch still keeps as many cores busy as the fused batches did
            const uint32_t looped_cores = div_up(p.Mt, looped->per_core_M) * div_up(p.Nt, looped->per_core_N);
            if (looped_cores >= chosen->cores) {
                chosen->blocking = subblock_->subblock(p, chosen->family, *looped);
                chosen->fuse_batch = false;
                chosen->cores = looped_cores;
            }
        }
    }
    std::vector<Candidate> result = {*chosen};
    for (auto& n : k_depth_neighbours(p, hw, *chosen)) {
        result.push_back(std::move(n));
    }
    return result;
}

std::vector<Candidate> FactoryBlockingSource::candidates(const MatmulDesc& p, const HardwareDesc& hw) const {
    if (sharded_layout(p)) {
        return sharded_candidates(p, hw);
    }
    std::vector<Candidate> result;
    const uint32_t cores = hw.grid.x * hw.grid.y;
    // On a sub-device the configs name its cores; the factories otherwise start at (0, 0)
    const std::optional<CoreRange> workers = pinned_workers(hw);
    auto add = [&](Family family, std::optional<Blocking> b, bool fuse_batch) {
        if (!b) {
            return;
        }
        b = subblock_->subblock(p, family, *b);
        result.push_back({family, *b, cores_used(p, hw, family, *b, fuse_batch), hw.grid, workers, false, fuse_batch});
    };
    const bool mcast_ok = !needs_single_k_reuse(p);
    if (broadcasts_a(p)) {
        if (mcast_ok && !no_mcast_1d(p)) {
            add(Family::Mcast1DIn1,
                blocking_->block(p, hw, Family::Mcast1DIn1, {div_up(p.Mt, cores), p.Nt, false}, {}),
                false);
        }
        return result;
    }
    // Batched B can't be fused into M: the mcast families then loop over the batch. Neither can a transposed A
    // of multi-row batch matrices (its reads would cross batch boundaries).
    const bool fuse_batch = p.batch_b == 1 && !(p.transpose_a && p.batch_a > 1 && p.Mt > 1);
    // Reuse needs matching batches: batched B, or (when the mcast kernels can't run it) a single batch
    if (p.batch_a == p.batch_b && (p.batch_b > 1 || !mcast_ok)) {
        add(Family::Reuse, reuse_blocking(p, hw), true);
    }
    if (!mcast_ok) {
        return result;
    }
    const uint32_t M = output_rows(p, fuse_batch);
    add(Family::Mcast2D,
        blocking_->block(p, hw, Family::Mcast2D, {div_up(M, hw.grid.y), div_up(p.Nt, hw.grid.x), fuse_batch}, {}),
        fuse_batch);
    if (!no_mcast_1d(p)) {
        add(Family::Mcast1DIn0,
            blocking_->block(p, hw, Family::Mcast1DIn0, {M, div_up(p.Nt, cores), fuse_batch}, {}),
            fuse_batch);
        add(Family::Mcast1DIn1,
            blocking_->block(p, hw, Family::Mcast1DIn1, {div_up(M, cores), p.Nt, fuse_batch}, {}),
            fuse_batch);
    }
    return result;
}

// Reuse (batched B): per_core_N = Nt and per_core_M is the tallest slice of a batch matrix that still gives
// every core a block (all of Mt when the batch alone fills the grid) and fits L1. Block-float B with A tiles under
// 16 rows needs a single K block.
std::optional<Blocking> FactoryBlockingSource::reuse_blocking(const MatmulDesc& p, const HardwareDesc& hw) const {
    const uint32_t cores = hw.grid.x * hw.grid.y;
    const BlockRules rules{.k_fixed = needs_single_k_reuse(p) ? p.Kt : 0};
    for (uint32_t per_core_M : divisors_desc(p.Mt)) {
        const bool fills_grid = p.batch_a * (p.Mt / per_core_M) >= cores;
        if (!fills_grid && per_core_M > 1) {
            continue;
        }
        if (auto b = blocking_->block(p, hw, Family::Reuse, {per_core_M, p.Nt, true}, rules)) {
            return b;
        }
    }
    return std::nullopt;
}

// A sharded operand or output fixes the family, grid and per-core sizes; returns that layout's candidate, or
// nothing when the combination isn't supported or doesn't fit.
std::vector<Candidate> FactoryBlockingSource::sharded_candidates(const MatmulDesc& p, const HardwareDesc& hw) const {
    std::vector<Candidate> result;
    auto add = [&](Family family,
                   std::optional<Blocking> b,
                   CoreCoord grid,
                   std::optional<CoreRange> workers,
                   bool transpose_mcast) {
        if (!b) {
            return;
        }
        b = subblock_->subblock(p, family, *b);
        HardwareDesc sub = hw;
        sub.grid = grid;
        result.push_back({family, *b, cores_used(p, sub, family, *b, true), grid, workers, transpose_mcast, true});
    };
    const uint32_t M = p.batch_a * p.Mt;  // batch fused into M (the sharded layouts require it)
    const bool b_batched = p.batch_b > 1;

    if (p.a.sharded()) {
        const auto& a = p.a;
        if (!a.in_l1 || !a.has_shard_spec) {
            return result;
        }
        // A sharded output must be L1 and laid out like A
        if (p.out.sharded() && (!p.out.in_l1 || p.out.layout != a.layout)) {
            return result;
        }
        const CoreCoord grid = a.shard_grid.grid_size();
        const BlockRules out_rules{.sharded_out = p.out.sharded()};
        if (a.layout == MemoryLayout::WidthSharded) {
            // 1D in0-mcast on A's grid: each core holds all of M for a slice of K, and computes a slice of N
            if (b_batched || p.b.l1_sharded() || a.col_major || a.shard_h != M) {
                return result;
            }
            BlockRules rules = out_rules;
            rules.k_divides = a.shard_w;
            rules.k_preferred = a.shard_w;
            const auto per_core_N = div_up(p.Nt, a.shard_cores);
            add(Family::Mcast1DIn0,
                blocking_->block(p, hw, Family::Mcast1DIn0, {M, per_core_N, true}, rules),
                grid,
                a.shard_grid,
                false);
        } else if (a.layout == MemoryLayout::HeightSharded) {
            // Each core holds a slice of M over all of K
            if (a.col_major || a.shard_w != p.Kt) {
                return result;
            }
            if (b_batched) {
                // Reuse on A's grid; B interleaved, or sharded exactly like A
                if (p.b.l1_sharded() && !p.b_shard_matches_a) {
                    return result;
                }
                const bool divides = p.Mt % a.shard_h == 0;
                const bool whole_batches = a.shard_h % p.Mt == 0 && M % a.shard_h == 0;
                if (!divides && !whole_batches) {
                    return result;
                }
                add(Family::Reuse,
                    blocking_->block(p, hw, Family::Reuse, {a.shard_h, p.Nt, true}, {.k_fixed = p.Kt}),
                    grid,
                    std::nullopt,
                    false);
            } else {
                // 1D in1-mcast on A's grid, reading A in place (in0_block_w = K)
                if (p.b.l1_sharded()) {
                    return result;
                }
                BlockRules rules = out_rules;
                rules.k_fixed = p.Kt;
                add(Family::Mcast1DIn1,
                    blocking_->block(p, hw, Family::Mcast1DIn1, {a.shard_h, p.Nt, true}, rules),
                    grid,
                    a.shard_grid,
                    false);
            }
        } else if (a.layout == MemoryLayout::BlockSharded) {
            // 2D on A's grid: rows of cores split M, columns split K (for A) and N (for the output). On a single
            // row or column of cores one of those splits is trivial.
            if (b_batched || p.b.l1_sharded()) {
                return result;
            }
            const uint32_t virtual_x = a.col_major ? grid.y : grid.x;
            const uint32_t virtual_y = a.col_major ? grid.x : grid.y;
            const uint32_t per_core_M = div_up(M, virtual_y);
            const uint32_t per_core_N = div_up(p.Nt, virtual_x);
            if (per_core_M != a.shard_h || div_up(p.Kt, a.shard_w) != virtual_x || div_up(M, a.shard_h) != virtual_y) {
                return result;
            }
            BlockRules rules = out_rules;
            // in0_block_w must divide the shard width when the shards tile K exactly; otherwise 1
            if (p.Kt % a.shard_w == 0) {
                rules.k_divides = a.shard_w;
                rules.k_preferred = a.shard_w;
            } else {
                rules.k_fixed = 1;
            }
            add(Family::Mcast2D,
                blocking_->block(p, hw, Family::Mcast2D, {per_core_M, per_core_N, true}, rules),
                grid,
                a.shard_grid,
                a.col_major);
        }
        return result;
    }

    // Interleaved inputs, sharded output: the output layout picks the family. A given output shard spec also
    // fixes the grid and per-core sizes; without one the layout covers the grid and matmul derives the spec.
    const auto& out = p.out;
    if (!out.in_l1 || b_batched || out.col_major) {
        return result;
    }
    const uint32_t cores = hw.grid.x * hw.grid.y;
    const BlockRules rules{.sharded_out = true};
    if (out.has_shard_spec) {
        const CoreCoord grid = out.shard_grid.grid_size();
        // A shard larger than the output holds all of it; per-core sizes stop at the output's
        const uint32_t shard_h = std::min(out.shard_h, M);
        const uint32_t shard_w = std::min(out.shard_w, p.Nt);
        const auto fits = [&](uint32_t rows, uint32_t cols) {
            return div_up(M, shard_h) == rows && div_up(p.Nt, shard_w) == cols;
        };
        // A block-sharded output on a row (column) of cores is width (height) sharded; on one core it stays 2D
        const bool block = out.layout == MemoryLayout::BlockSharded;
        const bool row_of_cores = out.layout == MemoryLayout::WidthSharded || (block && grid.y == 1 && grid.x > 1);
        const bool col_of_cores = out.layout == MemoryLayout::HeightSharded || (block && grid.x == 1 && grid.y > 1);
        if (row_of_cores && fits(1, out.shard_cores)) {
            add(Family::Mcast1DIn0,
                blocking_->block(p, hw, Family::Mcast1DIn0, {M, shard_w, true}, rules),
                grid,
                out.shard_grid,
                false);
        } else if (col_of_cores && fits(out.shard_cores, 1)) {
            add(Family::Mcast1DIn1,
                blocking_->block(p, hw, Family::Mcast1DIn1, {shard_h, p.Nt, true}, rules),
                grid,
                out.shard_grid,
                false);
        } else if (block && fits(grid.y, grid.x)) {
            add(Family::Mcast2D,
                blocking_->block(p, hw, Family::Mcast2D, {shard_h, shard_w, true}, rules),
                grid,
                out.shard_grid,
                false);
        }
        if (result.empty()) {
            // The spec's grid doesn't match the output (e.g. one core for several batches' worth of shards).
            // matmul rebuilds the output spec from the config, so keep the shard shape and derive the grid.
            const uint32_t rows = div_up(M, shard_h);
            const uint32_t cols = div_up(p.Nt, shard_w);
            const auto workers = pinned_workers(hw);
            if (block && rows <= hw.grid.y && cols <= hw.grid.x) {
                add(Family::Mcast2D,
                    blocking_->block(p, hw, Family::Mcast2D, {shard_h, shard_w, true}, rules),
                    hw.grid,
                    workers,
                    false);
            } else if (out.layout != MemoryLayout::HeightSharded && rows == 1 && cols <= cores) {
                add(Family::Mcast1DIn0,
                    blocking_->block(p, hw, Family::Mcast1DIn0, {M, shard_w, true}, rules),
                    hw.grid,
                    workers,
                    false);
            } else if (out.layout != MemoryLayout::WidthSharded && cols == 1 && rows <= cores) {
                add(Family::Mcast1DIn1,
                    blocking_->block(p, hw, Family::Mcast1DIn1, {shard_h, p.Nt, true}, rules),
                    hw.grid,
                    workers,
                    false);
            }
        }
    } else if (out.layout == MemoryLayout::WidthSharded) {
        add(Family::Mcast1DIn0,
            blocking_->block(p, hw, Family::Mcast1DIn0, {M, div_up(p.Nt, cores), true}, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    } else if (out.layout == MemoryLayout::HeightSharded) {
        add(Family::Mcast1DIn1,
            blocking_->block(p, hw, Family::Mcast1DIn1, {div_up(M, cores), p.Nt, true}, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    } else if (out.layout == MemoryLayout::BlockSharded) {
        add(Family::Mcast2D,
            blocking_->block(p, hw, Family::Mcast2D, {div_up(M, hw.grid.y), div_up(p.Nt, hw.grid.x), true}, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    }
    return result;
}

std::vector<Candidate> k_depth_neighbours(const MatmulDesc& p, const HardwareDesc& hw, const Candidate& c) {
    std::vector<Candidate> result;
    if (sharded_layout(p)) {
        return result;
    }
    auto divisors = divisors_desc(p.Kt);  // largest first
    auto legal_at = [&](uint32_t k) -> std::optional<Candidate> {
        Candidate n = c;
        n.blocking.in0_block_w = k;
        if (factory_limit_error(p, hw, to_program_config(p, n)).empty()) {
            return n;
        }
        return std::nullopt;
    };
    const auto here = std::find(divisors.begin(), divisors.end(), c.blocking.in0_block_w);
    if (here == divisors.end()) {
        return result;
    }
    // Deeper: the divisors before `here`, nearest first; shallower: those after it
    for (auto it = std::make_reverse_iterator(here); it != divisors.rend(); ++it) {
        if (auto n = legal_at(*it)) {
            result.push_back(*n);
            break;
        }
    }
    for (auto it = std::next(here); it != divisors.end(); ++it) {
        if (auto n = legal_at(*it)) {
            result.push_back(*n);
            break;
        }
    }
    return result;
}

}  // namespace ttnn::operations::matmul::auto_config
