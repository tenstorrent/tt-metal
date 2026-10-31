// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_candidates.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <utility>

#include <tt_stl/assert.hpp>

// Ported from the default-config selector proposed in #58143: its candidate enumeration (K depths, output blocks,
// subblock variants, layouts), its blocking, subblock and family heuristics, the roofline the family heuristic ranks
// with, its L1 limit check and the factories' circular buffer sizes, with the paths for sharded tensors, sub-devices,
// global CBs and tiles other than 32x32 removed (the selector only runs on interleaved 32x32-tile matmuls), and
// Blackhole's parameters (FactoryBlockingSource BH overrides) fixed. The blocking rules a sharded layout imposes
// (BlockRules) are therefore always the defaults and are dropped.
namespace ttnn::operations::matmul::model {

namespace {

constexpr std::uint32_t kMcastBuffering = 2;  // MCAST_INPUT_BUFFERING_DEPTH

// HeuristicBlocking::Params::for_arch(BLACKHOLE) and HeuristicFamily::Params
constexpr std::uint32_t kMinIn0BlockW = 2;
constexpr std::uint32_t kMaxIn0BlockW = 8;
constexpr std::uint32_t kMaxCostlyKBlocks = 8;
constexpr std::uint32_t kMaxCostlyIn0BlockW = 32;
constexpr std::uint32_t kMaxSelfReadTilesPerKStep = 12;  // 8 on Wormhole
constexpr bool kKDepthOverBlockSize = true;              // false on Wormhole
constexpr double kReuseBlockCycles = 32000;
constexpr double kReuseWriteBytesPerCycle = 3;
constexpr double kReuseSwitchMargin = 1.25;
constexpr double kOneDCoreAdvantage = 1.5;
constexpr std::uint32_t kBlocksPerK = 3;  // EnumeratingSource::blocks_per_k

std::uint32_t div_up(std::uint32_t a, std::uint32_t b) { return (a + b - 1) / b; }
std::uint32_t align_up(std::uint32_t x, std::uint32_t a) { return div_up(x, a) * a; }

bool is_block_float(DataType t) { return t == DataType::kBfloat8B || t == DataType::kBfloat4B; }

// Each of `parts` cores' share of N
std::uint32_t split_n(const MatmulDesc& p, std::uint32_t parts) { return div_up(p.Nt, parts); }

// A batch of one against a batched B: only 1D in1-mcast runs it (A resident, looping over B's batches)
bool broadcasts_a(const MatmulDesc& p) { return p.batch_a == 1 && p.batch_b > 1; }

std::uint32_t output_rows(const MatmulDesc& p, bool fuse_batch) { return fuse_batch ? p.batch_a * p.Mt : p.Mt; }

std::uint32_t num_cores(const HardwareDesc& hw) { return hw.grid_x * hw.grid_y; }

std::uint32_t cores_used(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse) {
    if (family == Family::kReuse) {
        return std::min(p.batch_a * p.Mt / b.per_core_M, num_cores(hw));
    }
    return div_up(output_rows(p, fuse), b.per_core_M) * div_up(p.Nt, b.per_core_N);
}

// Divisors of n, largest first
std::vector<std::uint32_t> divisors_desc(std::uint32_t n) {
    std::vector<std::uint32_t> small, large;
    for (std::uint32_t d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            small.push_back(d);
            if (d != n / d) {
                large.push_back(n / d);
            }
        }
    }
    large.insert(large.end(), small.rbegin(), small.rend());
    return large;
}

// Tiles held in DST for one subblock. Half-sync even with dst_full_sync: the 2D, 1D and Reuse factories compute wrong
// values above the half-sync count, and the device op's validation holds them to it (#59689).
std::uint32_t max_subblock_area(const MatmulDesc& p, Family family) {
    std::uint32_t area = 8;
    if (p.fp32_dest_acc) {
        area /= 2;
        if (family == Family::kReuse) {
            area = std::min(area, 4u);  // the reuse factory caps fp32-accumulating subblocks at 4
        }
    }
    return area;
}

// ---- circular buffers (factory/matmul_buffers.cpp, interleaved tensors) ----

std::uint32_t entry_size(std::uint32_t bytes, const HardwareDesc& hw) { return align_up(bytes, hw.dram_alignment); }

// Buffer sizes are summed in 64 bits: block sizes from huge K or N can exceed 4 GB, which must not wrap into a size
// that fits L1.
// in0 entries' bytes (the in0_transposed copy has the same size) plus the out / partials buffers
std::uint64_t finish(const MatmulDesc& p, std::uint64_t in0_bytes, std::uint32_t k, std::uint64_t out_block_tiles) {
    const bool l1_acc = p.packer_l1_acc && p.Kt / k > 1;
    const DataType interm = p.fp32_dest_acc ? DataType::kFloat32 : (l1_acc ? DataType::kBfloat16 : p.out);
    std::uint64_t total = std::uint64_t{tile_size_bytes(p.out)} * out_block_tiles;
    if (interm != p.out) {
        total += std::uint64_t{tile_size_bytes(interm)} * out_block_tiles;  // partials can't share the output buffer
    }
    if (p.transpose_a) {
        total += in0_bytes;  // in0_transposed
    }
    return total;
}

std::uint64_t cb_bytes(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse) {
    const std::uint64_t a_batches = fuse ? 1 : p.batch_a;
    const std::uint64_t b_batches = fuse ? 1 : p.batch_b;
    const std::uint64_t nb = p.Kt / b.in0_block_w;
    const std::uint64_t t0 = entry_size(tile_size_bytes(p.in0), hw);
    const std::uint64_t t1 = entry_size(tile_size_bytes(p.in1), hw);
    const std::uint64_t bias =
        p.bias_tile_bytes != 0 ? std::uint64_t{entry_size(p.bias_tile_bytes, hw)} * b.out_block_w : 0;
    const std::uint32_t k = b.in0_block_w;
    const std::uint64_t out_block_tiles = std::uint64_t{b.out_block_h} * b.out_block_w;
    switch (family) {
        case Family::k2D: {
            const std::uint64_t buf = a_batches * nb > 1 ? kMcastBuffering : 1;
            const std::uint64_t in0 = t0 * b.out_block_h * k * buf;
            return in0 + t1 * b.out_block_w * k * buf + bias + finish(p, in0, k, out_block_tiles);
        }
        case Family::k1DMcastIn0: {
            const std::uint64_t in0 = t0 * b.out_block_h * k * (a_batches * nb > 1 ? kMcastBuffering : 1);
            const std::uint64_t in1 = t1 * b.out_block_w * k * (b_batches * nb > 1 ? kMcastBuffering : 1);
            return in0 + in1 + bias + finish(p, in0, k, out_block_tiles);
        }
        case Family::k1DMcastIn1: {
            std::uint64_t in0_entries = std::uint64_t{b.out_block_h} * k;
            if (a_batches == 1 && b_batches > 1) {
                in0_entries = std::uint64_t{b.per_core_M} * nb * k;  // A resident across B's batches
            } else if (a_batches * nb > 1) {
                in0_entries *= 2;
            }
            const std::uint64_t in0 = t0 * in0_entries;
            const std::uint64_t in1 = t1 * b.out_block_w * k * (b_batches * nb > 1 ? 2 : 1);
            return in0 + in1 + bias + finish(p, in0, k, out_block_tiles);
        }
        case Family::kReuse: {
            const std::uint64_t pm_batch = b.per_core_M > p.Mt ? p.Mt : b.per_core_M;
            const std::uint64_t in0 = t0 * pm_batch * k * 2;
            const std::uint64_t in1 = t1 * b.per_core_N * k * 2;
            // the whole per-batch [M, N] bias block, unaligned
            const std::uint64_t reuse_bias = p.bias_tile_bytes * pm_batch * b.per_core_N;
            return in0 + in1 + reuse_bias + finish(p, in0, k, std::uint64_t{b.per_core_M} * b.per_core_N);
        }
    }
    TT_THROW("Unreachable");
}

bool fits(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse) {
    return cb_bytes(p, hw, family, b, fuse) <= hw.l1_cb_budget;
}

// factory_limit_error, for the configs the enumerator emits
bool factory_limit_ok(const MatmulDesc& p, const HardwareDesc& hw, const Candidate& c) {
    Blocking b = c.blocking;
    if (b.in0_block_w == 0 || b.per_core_M == 0 || b.per_core_N == 0 || b.out_subblock_h == 0 ||
        b.out_subblock_w == 0) {
        return false;
    }
    const bool reuse = c.family == Family::kReuse;
    if (reuse) {
        b.out_block_h = b.per_core_M;
        b.out_block_w = b.per_core_N;
    } else if (b.out_block_h == 0 || b.out_block_w == 0) {
        return false;
    }
    if (b.out_subblock_h * b.out_subblock_w > max_subblock_area(p, c.family)) {
        return false;
    }
    return fits(p, hw, c.family, b, reuse ? true : c.fuse_batch);
}

// ---- roofline (config/roofline_estimator.cpp) ----

double roofline_cycles(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse) {
    const double fid = fidelity_phases(p.fidelity);
    const double a_bytes = tile_size_bytes(p.in0);
    const double b_bytes = tile_size_bytes(p.in1);
    const double cycles_per_product = 2.0 * kTileDim * kTileDim * kTileDim * fid / (hw.matmul_flops_per_cycle * 1.0);
    const double Kt = p.Kt;
    double products = 0, received = 0, a_total = 0, b_total = 0;
    if (family == Family::kReuse) {
        const double cores = num_cores(hw);
        const double blocks = std::ceil(std::ceil(double(p.batch_a) * p.Mt / b.per_core_M) / cores);
        const double batches_per_block = std::max(1.0, double(b.per_core_M) / p.Mt);
        products = blocks * b.per_core_M * p.Nt * Kt;
        received = blocks * (b.per_core_M * Kt * a_bytes + batches_per_block * Kt * p.Nt * b_bytes);
        a_total = double(p.batch_a) * p.Mt * Kt;
        b_total = double(p.batch_b) * Kt * p.Nt * div_up(p.Mt, b.per_core_M);
    } else {
        const double loops = fuse ? 1.0 : std::max(p.batch_a, p.batch_b);
        const double a_passes = double(b.per_core_N) / b.out_block_w;
        const double b_passes = double(b.per_core_M) / b.out_block_h;
        products = double(b.per_core_M) * b.per_core_N * Kt * loops;
        received = b.per_core_M * Kt * a_passes * (broadcasts_a(p) ? 1.0 : loops) * a_bytes +
                   b.per_core_N * Kt * b_passes * loops * b_bytes;
        a_total = double(p.batch_a) * p.Mt * Kt * a_passes;
        b_total = (p.batch_b > 1 ? double(p.batch_b) : loops) * Kt * p.Nt * b_passes;
    }
    const double out_total = double(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
    const double dram = (p.a.in_l1 ? 0.0 : a_total * a_bytes) + (p.b.in_l1 ? 0.0 : b_total * b_bytes) +
                        (p.out_place.in_l1 ? 0.0 : out_total * tile_size_bytes(p.out));
    return std::max({products * cycles_per_product, received / hw.noc_bytes_per_cycle, dram / hw.dram_bytes_per_cycle});
}

// ---- HeuristicBlocking ----

// A layout's per-core work: per_core_M x per_core_N output tiles, with the batch folded into M or looped over
struct Split {
    std::uint32_t per_core_M = 0, per_core_N = 0;
    bool fuse_batch = true;
    bool operator==(const Split&) const = default;
};

bool costly_k_blocks(const MatmulDesc& p, std::uint32_t h, std::uint32_t w) {
    return is_block_float(p.in0) || is_block_float(p.in1) || (!p.packer_l1_acc && std::min(h, w) >= 2);
}

std::uint32_t max_in0_block_w(const MatmulDesc& p, Family family, std::uint32_t h, std::uint32_t w) {
    std::uint32_t self_read = 0;
    switch (family) {
        case Family::k2D: self_read = 0; break;
        case Family::k1DMcastIn0: self_read = w; break;
        case Family::k1DMcastIn1: self_read = broadcasts_a(p) ? 0 : h; break;
        case Family::kReuse: self_read = h + w; break;
    }
    std::uint32_t depth = kMaxIn0BlockW;
    if (family == Family::k2D && costly_k_blocks(p, h, w)) {
        depth = std::max(depth, std::min(div_up(p.Kt, kMaxCostlyKBlocks), kMaxCostlyIn0BlockW));
    }
    const std::uint32_t self_read_limit =
        self_read == 0 ? depth : std::max(kMinIn0BlockW, kMaxSelfReadTilesPerKStep / self_read);
    return std::min({depth, p.Kt, self_read_limit});
}

std::optional<Blocking> block_2d(const MatmulDesc& p, const HardwareDesc& hw, const Split& s) {
    const std::uint32_t min_k = kKDepthOverBlockSize ? std::min(kMinIn0BlockW, p.Kt) : 1;
    struct Rank {
        bool deep_enough;
        std::uint64_t volume;
        std::int64_t traffic;
        auto tie() const { return std::tie(deep_enough, volume, traffic); }
    };
    auto rank_of = [&](std::uint32_t k, std::uint32_t h, std::uint32_t w) {
        const std::uint64_t area = static_cast<std::uint64_t>(h) * w;
        const bool count_partials = kKDepthOverBlockSize || costly_k_blocks(p, h, w);
        const std::int64_t traffic =
            static_cast<std::int64_t>(k) * (h + w) + (count_partials ? static_cast<std::int64_t>(area) : 0);
        return Rank{k >= min_k, area * k, -traffic};
    };
    std::optional<Blocking> best;
    std::optional<Rank> best_rank;
    for (std::uint32_t h : divisors_desc(s.per_core_M)) {
        for (std::uint32_t w : divisors_desc(s.per_core_N)) {
            const std::uint32_t k_max = max_in0_block_w(p, Family::k2D, h, w);
            if (best_rank && best_rank->deep_enough && static_cast<std::uint64_t>(h) * w * k_max < best_rank->volume) {
                break;
            }
            for (std::uint32_t k : divisors_desc(p.Kt)) {
                if (k > k_max) {
                    continue;
                }
                const Blocking b{s.per_core_M, s.per_core_N, k, h, w, 0, 0};
                if (!fits(p, hw, Family::k2D, b, s.fuse_batch)) {
                    continue;
                }
                const Rank rank = rank_of(k, h, w);
                if (!best_rank || rank.tie() > best_rank->tie()) {
                    best = b;
                    best_rank = rank;
                }
                break;
            }
        }
    }
    return best;
}

std::uint32_t widest_divisor_within(std::uint32_t n, std::uint32_t limit) {
    for (std::uint32_t d = std::min(n, limit); d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

std::optional<Blocking> block_1d(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Split& s) {
    const std::uint32_t per_core_M = s.per_core_M, per_core_N = s.per_core_N;
    const bool is_tall = family == Family::k1DMcastIn1;
    const std::uint32_t fixed_full = is_tall ? per_core_N : per_core_M;
    const std::uint32_t cheap_full = is_tall ? per_core_M : per_core_N;
    const std::uint32_t M_rows = output_rows(p, s.fuse_batch);
    const std::uint32_t max_area = max_subblock_area(p, family);
    const std::uint32_t split_w = widest_divisor_within(per_core_N, max_area);
    const bool split = !is_tall && per_core_N > max_area && split_w > 1;

    auto fit = [&](std::uint32_t out_block_h, std::uint32_t out_block_w) -> std::optional<Blocking> {
        if (is_tall && div_up(M_rows, per_core_M) == 1 && M_rows % out_block_h != 0 && per_core_M != out_block_h) {
            return std::nullopt;
        }
        if (split && out_block_w > split_w) {
            return std::nullopt;
        }
        const std::uint32_t k_limit = max_in0_block_w(p, family, out_block_h, out_block_w);
        for (std::uint32_t k : divisors_desc(p.Kt)) {
            if (k > k_limit) {
                continue;
            }
            const Blocking b{per_core_M, per_core_N, k, out_block_h, out_block_w, 0, 0};
            if (fits(p, hw, family, b, s.fuse_batch)) {
                return b;
            }
        }
        return std::nullopt;
    };
    auto area_of = [](const Blocking& b) { return static_cast<std::uint64_t>(b.out_block_h) * b.out_block_w; };

    std::optional<Blocking> best;
    std::uint64_t best_product = 0;
    for (std::uint32_t cheap : divisors_desc(cheap_full)) {
        const auto b = fit(is_tall ? cheap : fixed_full, is_tall ? fixed_full : cheap);
        if (b) {
            const std::uint64_t product = area_of(*b) * b->in0_block_w;
            if (product > best_product || (product == best_product && area_of(*b) > area_of(*best))) {
                best = b;
                best_product = product;
            }
        }
        if (best && (best->in0_block_w > 1 || best_product == static_cast<std::uint64_t>(per_core_M) * per_core_N)) {
            break;
        }
    }
    if (!best || best->in0_block_w > 1 || p.Kt == 1) {
        return best;
    }
    // Single-tile K steps: shrink the multicast dimension too
    auto rank_of = [&](const Blocking& b) {
        const std::uint64_t area = area_of(b);
        const int unit_dims =
            (b.in0_block_w == 1) + (b.out_block_h == 1 && per_core_M > 1) + (b.out_block_w == 1 && per_core_N > 1);
        return std::make_tuple(
            area * b.in0_block_w,
            -unit_dims,
            area,
            -std::abs(static_cast<std::int64_t>(b.out_block_h) - static_cast<std::int64_t>(b.out_block_w)));
    };
    std::optional<Blocking> alt;
    for (std::uint32_t fixed : divisors_desc(fixed_full)) {
        for (std::uint32_t cheap : divisors_desc(cheap_full)) {
            const auto b = fit(is_tall ? cheap : fixed, is_tall ? fixed : cheap);
            if (b && (!alt || rank_of(*b) > rank_of(*alt))) {
                alt = b;
            }
        }
    }
    return alt && alt->in0_block_w > 1 ? alt : best;
}

std::optional<Blocking> block_reuse(const MatmulDesc& p, const HardwareDesc& hw, const Split& s) {
    const std::uint32_t k_limit = max_in0_block_w(p, Family::kReuse, s.per_core_M, s.per_core_N);
    for (std::uint32_t k : divisors_desc(p.Kt)) {
        if (k > k_limit) {
            continue;
        }
        const Blocking b{s.per_core_M, s.per_core_N, k, s.per_core_M, s.per_core_N, 0, 0};
        if (fits(p, hw, Family::kReuse, b, true)) {
            return b;
        }
    }
    return std::nullopt;
}

struct Layout {
    Family family;
    Split split;
    bool operator==(const Layout&) const = default;
};

// HeuristicBlocking, recording every layout it is asked to block (EnumeratingSource's RecordingBlocking)
class Blocker {
public:
    std::optional<Blocking> block(const MatmulDesc& p, const HardwareDesc& hw, Family family, const Split& s) {
        const Layout layout{family, s};
        if (std::find(layouts_.begin(), layouts_.end(), layout) == layouts_.end()) {
            layouts_.push_back(layout);
        }
        switch (family) {
            case Family::k2D: return block_2d(p, hw, s);
            case Family::k1DMcastIn0:
            case Family::k1DMcastIn1: return block_1d(p, hw, family, s);
            case Family::kReuse: return block_reuse(p, hw, s);
        }
        TT_THROW("Unreachable");
    }
    const std::vector<Layout>& layouts() const { return layouts_; }

private:
    std::vector<Layout> layouts_;
};

// ---- HeuristicSubblock ----

std::pair<std::uint32_t, std::uint32_t> choose_subblock(
    std::uint32_t block_h,
    std::uint32_t block_w,
    std::uint32_t max_area,
    bool prefer_two_wide,
    std::uint32_t h_divides) {
    std::pair<std::uint32_t, std::uint32_t> best{1, 1};
    for (std::uint32_t h = 1; h <= std::min(block_h, max_area); ++h) {
        if (block_h % h != 0 || (h_divides != 0 && h_divides % h != 0)) {
            continue;
        }
        for (std::uint32_t w = std::min(block_w, max_area / h); w >= 1; --w) {
            if (block_w % w != 0) {
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
            break;
        }
    }
    return best;
}

Blocking heuristic_subblock(const MatmulDesc& p, Family family, Blocking b) {
    const bool reuse = family == Family::kReuse;
    const bool prefer_two_wide = tile_size_bytes(p.in1) >= tile_size_bytes(p.in0);
    const auto [h, w] = choose_subblock(
        reuse ? b.per_core_M : b.out_block_h,
        reuse ? b.per_core_N : b.out_block_w,
        max_subblock_area(p, family),
        prefer_two_wide,
        reuse ? p.Mt : 0);
    b.out_subblock_h = h;
    b.out_subblock_w = w;
    return b;
}

// ---- HeuristicFamily ----

std::uint32_t per_core_input_tiles(const Blocking& b) { return b.per_core_M + b.per_core_N; }

std::uint64_t total_input_tiles(const MatmulDesc& p, Family family, const Blocking& b) {
    const std::uint64_t a_tiles = static_cast<std::uint64_t>(p.batch_a) * p.Mt * p.Kt;
    const std::uint64_t b_tiles = static_cast<std::uint64_t>(p.batch_b) * p.Kt * p.Nt;
    if (family == Family::kReuse) {
        return a_tiles + b_tiles * div_up(p.Mt, b.per_core_M);
    }
    return a_tiles * (b.per_core_N / b.out_block_w) + b_tiles * (b.per_core_M / b.out_block_h);
}

const Candidate* choose_mcast(const Candidate* two_d, const Candidate* in0, const Candidate* in1) {
    if (!two_d) {
        return in0 ? in0 : in1;
    }
    const double threshold = kOneDCoreAdvantage * two_d->cores;
    if (in0 && in0->cores >= threshold && (!in1 || in0->cores >= in1->cores)) {
        return in0;
    }
    if (in1 && in1->cores >= threshold) {
        return in1;
    }
    if (in0 && in0->cores >= two_d->cores &&
        per_core_input_tiles(in0->blocking) < per_core_input_tiles(two_d->blocking)) {
        return in0;
    }
    return two_d;
}

std::optional<Candidate> choose_family(const MatmulDesc& p, std::span<const Candidate> all) {
    auto find = [&](Family family) -> const Candidate* {
        for (const auto& c : all) {
            if (c.family == family) {
                return &c;
            }
        }
        return nullptr;
    };
    const Candidate* mcast = choose_mcast(find(Family::k2D), find(Family::k1DMcastIn0), find(Family::k1DMcastIn1));
    if (const auto* reuse = find(Family::kReuse)) {
        if (mcast && (mcast->cores >= kOneDCoreAdvantage * reuse->cores ||
                      total_input_tiles(p, Family::kReuse, reuse->blocking) >=
                          kOneDCoreAdvantage * total_input_tiles(p, mcast->family, mcast->blocking))) {
            return *mcast;
        }
        return *reuse;
    }
    if (mcast) {
        return *mcast;
    }
    return std::nullopt;
}

std::optional<Candidate> heuristic_family(const MatmulDesc& p, const HardwareDesc& hw, std::span<const Candidate> all) {
    auto chosen = choose_family(p, all);
    const bool one_tile_2d = chosen && chosen->family == Family::k2D &&
                             std::min(chosen->blocking.per_core_M, chosen->blocking.per_core_N) == 1;
    if (!one_tile_2d) {
        return chosen;
    }
    std::uint32_t most_cores = 0;
    for (const auto& c : all) {
        most_cores = std::max(most_cores, c.cores);
    }
    auto key = [&](const Candidate& c) {
        const bool clearly_fewer_cores = c.cores * kOneDCoreAdvantage <= most_cores;
        return std::make_tuple(
            clearly_fewer_cores,
            roofline_cycles(p, hw, c.family, c.blocking, c.fuse_batch),
            static_cast<int>(c.family));
    };
    const auto best = std::min_element(
        all.begin(), all.end(), [&](const Candidate& x, const Candidate& y) { return key(x) < key(y); });
    if (p.batch_b > 1 && best->family != Family::kReuse) {
        auto steps = [&](const Candidate& c) -> std::uint64_t {
            const std::uint64_t k_blocks = div_up(p.Kt, c.blocking.in0_block_w);
            if (c.family == Family::kReuse) {
                return div_up(p.batch_a * p.Mt / c.blocking.per_core_M, num_cores(hw)) * k_blocks;
            }
            const std::uint64_t loops = c.fuse_batch ? 1 : std::max(p.batch_a, p.batch_b);
            return std::uint64_t{c.blocking.per_core_M / c.blocking.out_block_h} *
                   (c.blocking.per_core_N / c.blocking.out_block_w) * k_blocks * loops;
        };
        const double best_cycles = roofline_cycles(p, hw, best->family, best->blocking, best->fuse_batch);
        for (const auto& c : all) {
            if (c.family == Family::kReuse && !std::get<0>(key(c)) && steps(c) <= steps(*best) &&
                roofline_cycles(p, hw, c.family, c.blocking, c.fuse_batch) < kOneDCoreAdvantage * best_cycles) {
                return c;
            }
        }
    }
    return *best;
}

// ---- FactoryBlockingSource (interleaved) ----

std::optional<Blocking> reuse_blocking(const MatmulDesc& p, const HardwareDesc& hw, Blocker& blocker) {
    const std::uint32_t cores = num_cores(hw);
    if (p.bias_rows > 1) {
        return blocker.block(p, hw, Family::kReuse, {p.Mt, p.Nt, true});
    }
    std::optional<Blocking> filled;
    for (std::uint32_t per_core_M : divisors_desc(p.Mt)) {
        const bool fills_grid = p.batch_a * (p.Mt / per_core_M) >= cores;
        if (!fills_grid && per_core_M > 1) {
            continue;
        }
        if (auto b = blocker.block(p, hw, Family::kReuse, {per_core_M, p.Nt, true})) {
            filled = b;
            break;
        }
    }
    if (!filled) {
        return filled;
    }
    auto estimate = [&](const Blocking& b) {
        const double blocks_per_core = std::ceil(std::ceil(double(p.batch_a) * p.Mt / b.per_core_M) / cores);
        const double tail = double(b.per_core_M) * p.Nt * tile_size_bytes(p.out) / kReuseWriteBytesPerCycle;
        return blocks_per_core * kReuseBlockCycles + roofline_cycles(p, hw, Family::kReuse, b, true) + tail;
    };
    std::optional<Blocking> best;
    double best_cost = 0;
    for (std::uint32_t per_core_M : divisors_desc(p.Mt)) {
        if (auto b = blocker.block(p, hw, Family::kReuse, {per_core_M, p.Nt, true})) {
            const double c = estimate(*b);
            if (!best || c < best_cost) {
                best = b;
                best_cost = c;
            }
        }
    }
    return best_cost * kReuseSwitchMargin < estimate(*filled) ? best : filled;
}

// The blocked candidate of each family that can run the problem, in family order
std::vector<Candidate> family_candidates(const MatmulDesc& p, const HardwareDesc& hw, Blocker& blocker) {
    std::vector<Candidate> result;
    const std::uint32_t cores = num_cores(hw);
    auto add = [&](Family family, std::optional<Blocking> b, bool fuse) {
        if (!b) {
            return;
        }
        b = heuristic_subblock(p, family, *b);
        result.push_back({family, *b, cores_used(p, hw, family, *b, fuse), hw.grid_x, hw.grid_y, fuse});
    };
    if (broadcasts_a(p)) {
        add(Family::k1DMcastIn1, blocker.block(p, hw, Family::k1DMcastIn1, {div_up(p.Mt, cores), p.Nt, false}), false);
        return result;
    }
    // Batched B can't be fused into M; neither can a transposed A of multi-row batch matrices
    const bool fuse = p.batch_b == 1 && !(p.transpose_a && p.batch_a > 1 && p.Mt > 1);
    if (p.batch_a == p.batch_b && p.batch_b > 1) {
        add(Family::kReuse, reuse_blocking(p, hw, blocker), true);
    }
    // A bias of a whole [M, N] block fuses only into Reuse
    if (p.bias_rows > 1 && !result.empty()) {
        return result;
    }
    const std::uint32_t M = output_rows(p, fuse);
    add(Family::k2D, blocker.block(p, hw, Family::k2D, {div_up(M, hw.grid_y), split_n(p, hw.grid_x), fuse}), fuse);
    add(Family::k1DMcastIn0, blocker.block(p, hw, Family::k1DMcastIn0, {M, split_n(p, cores), fuse}), fuse);
    add(Family::k1DMcastIn1, blocker.block(p, hw, Family::k1DMcastIn1, {div_up(M, cores), p.Nt, fuse}), fuse);
    return result;
}

// FactoryBlockingSource::propose with the heuristic policies, including loop_over_batch_if_better
std::optional<Candidate> heuristic_choice(const MatmulDesc& p, const HardwareDesc& hw, Blocker& blocker) {
    const auto all = family_candidates(p, hw, blocker);
    auto chosen = heuristic_family(p, hw, all);
    if (!chosen) {
        return chosen;
    }
    Candidate& c = *chosen;
    const bool splits_only_m =
        c.family == Family::k1DMcastIn1 || (c.family == Family::k2D && c.blocking.per_core_N == 1);
    if (splits_only_m && c.fuse_batch && p.batch_a > 1) {
        const std::uint32_t rows = c.family == Family::k2D ? hw.grid_y : num_cores(hw);
        if (auto looped = blocker.block(p, hw, c.family, {div_up(p.Mt, rows), c.blocking.per_core_N, false})) {
            const std::uint32_t looped_cores = div_up(p.Mt, looped->per_core_M) * div_up(p.Nt, looped->per_core_N);
            if (looped_cores >= c.cores) {
                c.blocking = heuristic_subblock(p, c.family, *looped);
                c.fuse_batch = false;
                c.cores = looped_cores;
            }
        }
    }
    return chosen;
}

// ---- EnumeratingSource ----

std::vector<std::uint32_t> k_depths(const MatmulDesc& p) {
    const std::vector<std::uint32_t> allowed = divisors_desc(p.Kt);
    std::set<std::uint32_t> depths;
    for (std::uint64_t cap = 1;; cap *= 2) {
        const auto deepest = std::find_if(allowed.begin(), allowed.end(), [&](std::uint32_t k) { return k <= cap; });
        if (deepest != allowed.end()) {
            depths.insert(*deepest);
        }
        if (cap >= p.Kt) {
            break;
        }
    }
    return {depths.begin(), depths.end()};
}

std::vector<Blocking> output_blocks(
    const MatmulDesc& p, const HardwareDesc& hw, const Layout& layout, std::uint32_t k) {
    const Split& s = layout.split;
    const bool reuse = layout.family == Family::kReuse;
    const std::uint32_t rows = output_rows(p, s.fuse_batch);
    std::vector<Blocking> frontier;
    for (std::uint32_t h : reuse ? std::vector<std::uint32_t>{s.per_core_M} : divisors_desc(s.per_core_M)) {
        if (layout.family == Family::k1DMcastIn1 && div_up(rows, s.per_core_M) == 1 && rows % h != 0 &&
            h != s.per_core_M) {
            continue;
        }
        for (std::uint32_t w : reuse ? std::vector<std::uint32_t>{s.per_core_N} : divisors_desc(s.per_core_N)) {
            const Blocking b{s.per_core_M, s.per_core_N, k, h, w, 0, 0};
            if (fits(p, hw, layout.family, b, s.fuse_batch)) {
                const bool contained = std::any_of(
                    frontier.begin(), frontier.end(), [&](const Blocking& f) { return f.out_block_w >= w; });
                if (!contained) {
                    frontier.push_back(b);
                }
                break;
            }
        }
    }
    auto area = [](const Blocking& b) { return static_cast<std::uint64_t>(b.out_block_h) * b.out_block_w; };
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
        if (best != nullptr && chosen.size() < kBlocksPerK) {
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

std::vector<Blocking> subblock_variants(const MatmulDesc& p, Family family, const Blocking& heuristic) {
    const std::uint32_t limit = max_subblock_area(p, family);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shapes;
    for (std::uint32_t h : divisors_desc(heuristic.out_block_h)) {
        if (family == Family::kReuse && p.Mt % h != 0) {
            continue;
        }
        for (std::uint32_t w : divisors_desc(heuristic.out_block_w)) {
            if (h * w <= limit) {
                shapes.emplace_back(h, w);
            }
        }
    }
    if (shapes.empty()) {
        return {heuristic};
    }
    auto best = [&](auto better) { return *std::min_element(shapes.begin(), shapes.end(), better); };
    auto area = [](const auto& s) { return s.first * s.second; };
    auto skew = [](const auto& s) { return std::max(s.first, s.second) - std::min(s.first, s.second); };
    const std::pair<std::uint32_t, std::uint32_t> picks[] = {
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

std::optional<Layout> looped_over_batch(const MatmulDesc& p, const HardwareDesc& hw, const Layout& layout) {
    if (layout.family == Family::kReuse || !layout.split.fuse_batch || p.batch_a <= 1) {
        return std::nullopt;
    }
    Layout looped = layout;
    looped.split.fuse_batch = false;
    switch (layout.family) {
        case Family::k2D: looped.split.per_core_M = div_up(p.Mt, hw.grid_y); break;
        case Family::k1DMcastIn0: looped.split.per_core_M = p.Mt; break;
        case Family::k1DMcastIn1: looped.split.per_core_M = div_up(p.Mt, num_cores(hw)); break;
        case Family::kReuse: break;
    }
    return looped;
}

// The fields of the program config a candidate becomes (EnumeratingSource deduplicates on its text)
auto config_key(const Candidate& c) {
    const Blocking& b = c.blocking;
    const bool reuse = c.family == Family::kReuse;
    return std::make_tuple(
        static_cast<int>(c.family),
        c.grid_x,
        c.grid_y,
        b.in0_block_w,
        b.out_subblock_h,
        b.out_subblock_w,
        b.per_core_M,
        b.per_core_N,
        reuse ? 0u : b.out_block_h,
        reuse ? 0u : b.out_block_w,
        reuse ? false : c.fuse_batch);
}

}  // namespace

HardwareDesc HardwareDesc::blackhole(std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t l1_cb_budget) {
    HardwareDesc hw;
    hw.grid_x = grid_x;
    hw.grid_y = grid_y;
    hw.l1_cb_budget = l1_cb_budget;
    return hw;
}

std::string unsupported_reason(const MatmulDesc& p) {
    if (p.a.sharded || p.b.sharded || p.out_place.sharded) {
        return "sharded tensor";
    }
    if (p.Mt == 0 || p.Kt == 0 || p.Nt == 0) {
        return "empty matmul";
    }
    if (p.batch_b > 1 && p.batch_a != p.batch_b && (p.batch_a != 1 || p.rank_a != p.rank_b || p.rank_a < 3)) {
        return "A and B batches don't match";
    }
    return "";
}

std::vector<Candidate> enumerate_candidates_with_narrower_grid(const MatmulDesc& p, const HardwareDesc& hw) {
    std::vector<Candidate> candidates = enumerate_candidates(p, hw);
    if (hw.grid_x > 1) {
        HardwareDesc narrower = hw;
        narrower.grid_x -= 1;
        const std::vector<Candidate> more = enumerate_candidates(p, narrower);
        candidates.insert(candidates.end(), more.begin(), more.end());
    }
    return candidates;
}

std::uint32_t circular_buffer_bytes(const MatmulDesc& p, const HardwareDesc& hw, const Candidate& c) {
    Blocking b = c.blocking;
    if (c.family == Family::kReuse) {
        b.out_block_h = b.per_core_M;
        b.out_block_w = b.per_core_N;
    }
    if (b.in0_block_w == 0) {
        return 0;
    }
    // Saturate: a config whose buffers exceed 4 GB never fits a budget
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        cb_bytes(p, hw, c.family, b, c.family == Family::kReuse ? true : c.fuse_batch),
        std::numeric_limits<std::uint32_t>::max()));
}

std::vector<Candidate> enumerate_candidates(const MatmulDesc& p, const HardwareDesc& hw) {
    if (!unsupported_reason(p).empty() || hw.grid_x == 0 || hw.grid_y == 0) {
        return {};
    }
    Blocker blocker;
    const std::optional<Candidate> chosen = heuristic_choice(p, hw, blocker);

    std::vector<Candidate> result;
    std::set<decltype(config_key(std::declval<Candidate>()))> seen;
    auto add = [&](const Candidate& c) {
        if (factory_limit_ok(p, hw, c) && seen.insert(config_key(c)).second) {
            result.push_back(c);
        }
    };
    if (chosen) {
        add(*chosen);
    }
    std::vector<Layout> layouts;
    for (const auto& layout : blocker.layouts()) {
        layouts.push_back(layout);
        if (auto looped = looped_over_batch(p, hw, layout);
            looped && std::find(layouts.begin(), layouts.end(), *looped) == layouts.end()) {
            layouts.push_back(*looped);
        }
    }
    for (const auto& layout : layouts) {
        for (std::uint32_t k : k_depths(p)) {
            for (const auto& b : output_blocks(p, hw, layout, k)) {
                for (const auto& v : subblock_variants(p, layout.family, heuristic_subblock(p, layout.family, b))) {
                    add(
                        {layout.family,
                         v,
                         cores_used(p, hw, layout.family, v, layout.split.fuse_batch),
                         hw.grid_x,
                         hw.grid_y,
                         layout.split.fuse_batch});
                }
            }
        }
    }
    return result;
}

Candidate make_candidate(
    const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch) {
    const bool fuse = family == Family::kReuse ? true : fuse_batch;
    return {family, b, cores_used(p, hw, family, b, fuse), hw.grid_x, hw.grid_y, fuse};
}

MatmulCandidate to_model_candidate(const Candidate& c) {
    MatmulCandidate m;
    m.family = c.family;
    m.grid_x = c.grid_x;
    m.grid_y = c.grid_y;
    m.in0_block_w = c.blocking.in0_block_w;
    m.out_subblock_h = c.blocking.out_subblock_h;
    m.out_subblock_w = c.blocking.out_subblock_w;
    m.per_core_M = c.blocking.per_core_M;
    m.per_core_N = c.blocking.per_core_N;
    if (c.family != Family::kReuse) {
        m.out_block_h = c.blocking.out_block_h;
        m.out_block_w = c.blocking.out_block_w;
        m.fuse_batch = c.fuse_batch;
    }
    m.transpose_mcast = false;
    return m;
}

}  // namespace ttnn::operations::matmul::model
