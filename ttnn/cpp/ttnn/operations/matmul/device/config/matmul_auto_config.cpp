// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

#include <algorithm>
#include <cmath>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul::auto_config {

namespace {

constexpr uint32_t TILE_DIM = 32;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
uint32_t align_up(uint32_t a, uint32_t alignment) { return div_up(a, alignment) * alignment; }

uint32_t tile_bytes(tt::DataFormat format, uint32_t h, uint32_t w) {
    return tt::tt_metal::Tile({h, w}).get_tile_size(format);
}
// Bytes of one A, B and output (or partials) tile. The factories size the output CBs with an
// in0_tile_h x in1_tile_w tile whatever the output tensor's tile.
uint32_t in0_tile_bytes(const Problem& p) { return tile_bytes(p.in0_format, p.in0_tile_h, TILE_DIM); }
uint32_t in1_tile_bytes(const Problem& p) { return tile_bytes(p.in1_format, TILE_DIM, p.in1_tile_w); }
uint32_t out_tile_bytes(const Problem& p, tt::DataFormat format) {
    return tile_bytes(format, p.in0_tile_h, p.in1_tile_w);
}

// The configs' worker cores on a sub-device (its rectangle); the factories otherwise start at (0, 0)
std::optional<CoreRange> pinned_workers(const HardwareDesc& hw) {
    if (!hw.pinned_origin) {
        return std::nullopt;
    }
    return CoreRange(hw.origin, CoreCoord(hw.origin.x + hw.grid.x - 1, hw.origin.y + hw.grid.y - 1));
}

// A batch of one against a batched B: only 1D in1-mcast can run it, keeping the core's rows of A resident
// in L1 and looping over B's batches (in0 reuse).
bool broadcasts_a(const Problem& p) { return p.batch_a == 1 && p.batch_b > 1; }

// Deepest in0_block_w the structure allows: with a single K block the mcast factories single-buffer the inputs,
// so reading the next block can't overlap math on the current one; keep two blocks when K allows it. The reuse
// factory always double-buffers. Beyond that, L1 decides.
uint32_t max_in0_block_w(uint32_t Kt, Family family) { return (family != Family::Reuse && Kt >= 2) ? Kt / 2 : Kt; }

// Rows of output tiles the mcast families split across cores: all batches when fused, else one batch.
uint32_t output_rows(const Problem& p, bool fuse_batch) { return fuse_batch ? p.batch_a * p.Mt : p.Mt; }

// Divisors of n, largest first.
std::vector<uint32_t> divisors_desc(uint32_t n) {
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    for (uint32_t d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            small.push_back(d);
            if (d != n / d) {
                large.push_back(n / d);
            }
        }
    }
    // large holds n/1, n/2, ... (descending); append the small ones in descending order
    large.insert(large.end(), small.rbegin(), small.rend());
    return large;
}

// Tiles held in the destination register for one subblock. Smaller tiles don't raise this: validation's
// tile-area dest count admits more of them, but subblocks above 8 tiles of 16-row tiles compute wrong values.
uint32_t max_subblock_area(const Problem& p, Family family) {
    uint32_t area = p.dst_full_sync_en ? 16 : 8;
    if (p.fp32_dest_acc_en) {
        area /= 2;
        // The reuse factory caps fp32-accumulating subblocks at 4 even with full-sync dest
        if (family == Family::Reuse) {
            area = std::min(area, 4u);
        }
    }
    return area;
}

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

bool packer_l1_acc_enabled(const Problem& p, Family /*family*/, uint32_t num_k_blocks) {
    if (!p.packer_l1_acc) {
        return false;
    }
    // Every factory accumulates in L1 whenever partials are kept between K blocks (#58178)
    return num_k_blocks > 1;
}

// Format partial sums are kept in between K blocks
tt::DataFormat interm_format(const Problem& p, Family family, uint32_t num_k_blocks) {
    if (p.fp32_dest_acc_en) {
        return tt::DataFormat::Float32;
    }
    return packer_l1_acc_enabled(p, family, num_k_blocks) ? tt::DataFormat::Float16_b : p.out_format;
}

bool is_block_float(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Bfp8:
        case tt::DataFormat::Bfp8_b:
        case tt::DataFormat::Bfp4:
        case tt::DataFormat::Bfp4_b:
        case tt::DataFormat::Bfp2:
        case tt::DataFormat::Bfp2_b: return true;
        default: return false;
    }
}

uint32_t k_depth_limit(const Problem& p, const EmpiricalDefaults& d, Family family) {
    return std::min(max_in0_block_w(p.Kt, family), d.max_in0_block_w);
}

}  // namespace

EmpiricalDefaults EmpiricalDefaults::for_arch(tt::ARCH /*arch*/) { return {}; }

HardwareDesc HardwareDesc::for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget) {
    HardwareDesc hw;
    hw.arch = arch;
    hw.grid = grid;
    hw.l1_cb_budget = l1_cb_budget;
    hw.dram_alignment = arch == tt::ARCH::BLACKHOLE ? 64 : 32;
    if (arch == tt::ARCH::BLACKHOLE) {
        hw.noc_bytes_per_cycle = 64;
        hw.dram_bytes_per_cycle = 512.0 / 1.35;  // 512 GB/s at 1.35 GHz
    }
    return hw;
}

uint32_t circular_buffer_bytes(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    // Sharded operands are read straight from L1, without DRAM-alignment padding
    const uint32_t in0_tile = p.a.sharded() ? in0_tile_bytes(p) : align_up(in0_tile_bytes(p), hw.dram_alignment);
    const uint32_t in1_tile = align_up(in1_tile_bytes(p), hw.dram_alignment);
    const uint32_t out_tile = out_tile_bytes(p, p.out_format);
    const uint32_t num_k_blocks = p.Kt / b.in0_block_w;

    const tt::DataFormat interm = interm_format(p, family, num_k_blocks);
    // interm shares the output's memory when formats match (and, for a sharded output, the block covers the
    // full per-core height); a sharded output's buffer is the output shard itself
    const bool interm_shares_out =
        interm == p.out_format && !(p.out.sharded() && family != Family::Reuse && b.per_core_M != b.out_block_h);

    uint32_t in0_bytes = 0;
    uint32_t in1_bytes = 0;
    uint32_t out_tiles = 0;
    uint32_t bias_bytes = 0;
    if (family == Family::Reuse) {
        const uint32_t per_batch_M = std::min(b.per_core_M, p.Mt);
        // A sharded A (and B sharded like it) is read in place
        in0_bytes = p.a.sharded() ? 0 : per_batch_M * b.in0_block_w * 2 * in0_tile;
        in1_bytes = p.b_shard_matches_a ? 0 : b.per_core_N * b.in0_block_w * 2 * in1_tile;
        out_tiles = b.per_core_M * b.per_core_N;
        bias_bytes = per_batch_M * b.per_core_N * p.bias_tile_bytes;
    } else {
        const bool looped_batches = !b.fuse_batch && p.batch_a > 1;
        const uint32_t buffering = (num_k_blocks > 1 || looped_batches) ? utilities::MCAST_INPUT_BUFFERING_DEPTH : 1;
        in0_bytes = b.out_block_h * b.in0_block_w * buffering * in0_tile;
        if (family == Family::Mcast1DIn1 && broadcasts_a(p) && !p.a.sharded()) {
            in0_bytes = b.per_core_M * p.Kt * in0_tile;  // A stays resident across B's batches
        }
        if (family == Family::Mcast1DIn1 && p.a.layout == Layout::HeightSharded) {
            // Read in place, unless the kernel has to extract sub-blocks from the shard into a copy
            const bool extract =
                num_k_blocks > 1 || (b.per_core_M / b.out_block_h > 1 && b.per_core_N / b.out_block_w > 1);
            in0_bytes = extract ? b.per_core_M * p.Kt * in0_tile : 0;
        }
        in1_bytes = b.out_block_w * b.in0_block_w * buffering * in1_tile;
        out_tiles = b.out_block_h * b.out_block_w;
        bias_bytes = p.bias_tile_bytes == 0 ? 0 : b.out_block_w * align_up(p.bias_tile_bytes, hw.dram_alignment);
    }
    uint32_t total = in0_bytes + in1_bytes + bias_bytes;
    if (p.out.sharded()) {
        total += b.per_core_M * b.per_core_N * out_tile;  // the output shard (allocated with the output)
    } else {
        total += out_tiles * out_tile;
    }
    if (!interm_shares_out) {
        total += out_tiles * out_tile_bytes(p, interm);
    }
    if (p.transpose_a) {
        // CB holding the transposed in0 block (the whole per-core A when A is read in place)
        total += in0_bytes != 0 ? in0_bytes : b.per_core_M * p.Kt * in0_tile;
    }
    return total;
}

namespace {

namespace {

// Extra conditions a layout puts on the blocking.
struct BlockRules {
    uint32_t k_divides = 0;    // in0_block_w must also divide this (a sharded A's shard width)
    uint32_t k_fixed = 0;      // in0_block_w must be exactly this
    bool sharded_out = false;  // out_block_w == per_core_N
    // in0_block_w to use when some block fits with it, even above the K depth limit: a width- or block-sharded
    // A's shard width. Each K block is then a whole shard column, multicast in place; a narrower one makes the
    // sender copy every K block out of the shard first (extract_shard_sub_blocks).
    uint32_t k_preferred = 0;

    bool prefers(uint32_t k) const { return k_preferred != 0 && k == k_preferred; }
    bool prefers_other(uint32_t k) const { return k_preferred != 0 && k != k_preferred; }
};

bool k_allowed(const BlockRules& rules, uint32_t k) {
    return (rules.k_fixed == 0 || k == rules.k_fixed) && (rules.k_divides == 0 || rules.k_divides % k == 0);
}

// Validation also admits out_block_h == 1 with a narrower out_block_w, but the 1D in0-mcast factory then
// writes a sharded output wrongly (#58046), so a sharded output's blocks always span per_core_N.
bool block_allowed(const BlockRules& rules, uint32_t per_core_N, uint32_t out_block_w) {
    return !rules.sharded_out || out_block_w == per_core_N;
}

}  // namespace

// 2D mcast (issue #57884 heuristic 1): largest in0_block_w * out_block_h * out_block_w that fits L1 (among
// blocks at the layout's preferred in0_block_w, if any fit), the work per K step. Ties go to the deeper K block
// (fewer steps), then the squarer output block (each loaded A and B tile is reused across the block's width and
// height, so a square block reuses the most for its area).
std::optional<Blocking> block_2d(
    const Problem& p,
    const HardwareDesc& hw,
    const EmpiricalDefaults& d,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const BlockRules& rules = {}) {
    const auto k_options = divisors_desc(p.Kt);
    std::optional<Blocking> best;
    uint64_t best_product = 0;
    auto skew = [](uint32_t h, uint32_t w) { return h > w ? h - w : w - h; };
    for (uint32_t h : divisors_desc(per_core_M)) {
        for (uint32_t w : divisors_desc(per_core_N)) {
            const uint64_t area = static_cast<uint64_t>(h) * w;
            const uint32_t k_max = rules.k_fixed != 0 ? rules.k_fixed : k_depth_limit(p, d, Family::Mcast2D);
            if (best && !rules.prefers_other(best->in0_block_w) &&
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
                Blocking b{per_core_M, per_core_N, k, h, w, 0, 0, fuse_batch};
                if (circular_buffer_bytes(p, hw, Family::Mcast2D, b) > hw.l1_cb_budget) {
                    continue;
                }
                const uint64_t product = area * k;
                const bool preference = best && rules.prefers(k) != rules.prefers(best->in0_block_w);
                if (preference
                        ? rules.prefers(k)
                        : (!best || product > best_product || (product == best_product && k > best->in0_block_w) ||
                           (product == best_product && k == best->in0_block_w &&
                            skew(h, w) < skew(best->out_block_h, best->out_block_w)))) {
                    best = b;
                    best_product = product;
                }
                break;  // largest fitting k for this output block
            }
        }
    }
    return best;
}

// 1D mcast (issue #57884 heuristic 2): keep the full per-core extent along the multicast dimension, shrink
// the other one only if needed; in0_block_w is the largest that fits (or the layout's preferred one, as in 2D).
// If keeping the full multicast extent only fits with single-tile K steps, both dimensions are searched as in
// 2D (largest in0_block_w * area; ties avoid 1-tile dimensions, then prefer the larger, squarer block).
std::optional<Blocking> block_1d(
    const Problem& p,
    const HardwareDesc& hw,
    const EmpiricalDefaults& d,
    Family family,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const BlockRules& rules = {}) {
    const bool is_tall = family == Family::Mcast1DIn1;
    const uint32_t fixed_full = is_tall ? per_core_N : per_core_M;
    const uint32_t cheap_full = is_tall ? per_core_M : per_core_N;
    const uint32_t M_rows = output_rows(p, fuse_batch);

    // The largest fitting in0_block_w for this output block, if any
    auto fit = [&](uint32_t out_block_h, uint32_t out_block_w) -> std::optional<Blocking> {
        // in1-mcast with a single block row: Mt % out_block_h == 0 or one output block per core
        if (is_tall && div_up(M_rows, per_core_M) == 1 && M_rows % out_block_h != 0 && per_core_M != out_block_h) {
            return std::nullopt;
        }
        if (!block_allowed(rules, per_core_N, out_block_w)) {
            return std::nullopt;
        }
        const uint32_t k_limit = rules.k_fixed != 0 ? rules.k_fixed : k_depth_limit(p, d, family);
        for (uint32_t k : divisors_desc(p.Kt)) {
            if ((k > k_limit && !rules.prefers(k)) || !k_allowed(rules, k)) {
                continue;
            }
            Blocking b{per_core_M, per_core_N, k, out_block_h, out_block_w, 0, 0, fuse_batch};
            if (circular_buffer_bytes(p, hw, family, b) <= hw.l1_cb_budget) {
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

// Reuse with a height-sharded A: each core computes its shard's rows against all of N, over all of K.
std::optional<Blocking> block_reuse_sharded(const Problem& p, const HardwareDesc& hw) {
    Blocking b{p.a.shard_h, p.Nt, p.Kt, p.a.shard_h, p.Nt, 0, 0};
    if (circular_buffer_bytes(p, hw, Family::Reuse, b) > hw.l1_cb_budget) {
        return std::nullopt;
    }
    return b;
}

// Reuse (batched B): per_core_N = Nt and per_core_M is the tallest slice of a batch matrix that still gives
// every core a block (all of Mt when the batch alone fills the grid) and fits L1; in0_block_w is the largest
// that fits within the K depth rule. Block-float B with A tiles under 16 rows needs a single K block: the
// factory computes wrong values when it splits K for those.
std::optional<Blocking> block_reuse(const Problem& p, const HardwareDesc& hw, const EmpiricalDefaults& d) {
    const uint32_t cores = hw.grid.x * hw.grid.y;
    const bool single_k_block = is_block_float(p.in1_format) && p.in0_tile_h < 16;
    for (uint32_t per_core_M : divisors_desc(p.Mt)) {
        const bool fills_grid = p.batch_a * (p.Mt / per_core_M) >= cores;
        if (!fills_grid && per_core_M > 1) {
            continue;
        }
        for (uint32_t k : divisors_desc(p.Kt)) {
            if (single_k_block ? k != p.Kt : k > k_depth_limit(p, d, Family::Reuse)) {
                continue;
            }
            Blocking b{per_core_M, p.Nt, k, per_core_M, p.Nt, 0, 0};
            if (circular_buffer_bytes(p, hw, Family::Reuse, b) <= hw.l1_cb_budget) {
                return b;
            }
        }
    }
    return std::nullopt;
}

uint32_t fidelity_multiplier(MathFidelity fidelity) {
    switch (fidelity) {
        case MathFidelity::LoFi: return 1;
        case MathFidelity::HiFi2: return 2;
        case MathFidelity::HiFi3: return 3;
        default: return 4;
    }
}

// Per-core roofline estimate, in cycles, of a blocked candidate: the largest of
//  - compute: the busiest core's tile products at the matrix engine's rate for the math fidelity (A tiles
//    shorter than 8 rows still take a full 8-row pass of the engine);
//  - NoC: the input bytes the busiest core receives (its rows of A and columns of B, once per output block
//    that uses them);
//  - DRAM: the input bytes read from DRAM in total (the mcast layouts read A once per output column block and
//    B once per output row block; Reuse reads A once and B once per M slice of a batch), over the chip's
//    bandwidth.
}  // namespace

RooflineTerms roofline(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    const double a_bytes = in0_tile_bytes(p);
    const double b_bytes = in1_tile_bytes(p);
    const double engine_share = std::min(p.in0_tile_h, 8u) / 8.0;
    const double cycles_per_product = 2.0 * p.in0_tile_h * TILE_DIM * p.in1_tile_w *
                                      fidelity_multiplier(p.math_fidelity) / (hw.matmul_flops_per_cycle * engine_share);
    const double Kt = p.Kt;
    double products = 0;  // per core
    double received = 0;  // bytes per core
    double a_total = 0;   // tiles read from memory
    double b_total = 0;
    if (family == Family::Reuse) {
        const double cores = hw.grid.x * hw.grid.y;
        const double blocks = std::ceil(std::ceil(double(p.batch_a) * p.Mt / b.per_core_M) / cores);
        const double batches_per_block = std::max(1.0, double(b.per_core_M) / p.Mt);
        products = blocks * b.per_core_M * p.Nt * Kt;
        received = blocks * (b.per_core_M * Kt * a_bytes + batches_per_block * Kt * p.Nt * b_bytes);
        a_total = double(p.batch_a) * p.Mt * Kt;
        b_total = double(p.batch_b) * Kt * p.Nt * div_up(p.Mt, b.per_core_M);
    } else {
        // Unfused, the layout loops over the batch; in0 reuse (a broadcast A) keeps A resident across it
        const double loops = b.fuse_batch ? 1.0 : std::max(p.batch_a, p.batch_b);
        const double a_passes = double(b.per_core_N) / b.out_block_w;
        const double b_passes = double(b.per_core_M) / b.out_block_h;
        products = double(b.per_core_M) * b.per_core_N * Kt * loops;
        received = b.per_core_M * Kt * a_passes * (broadcasts_a(p) ? 1.0 : loops) * a_bytes +
                   b.per_core_N * Kt * b_passes * loops * b_bytes;
        a_total = double(p.batch_a) * p.Mt * Kt * a_passes;
        b_total = (p.batch_b > 1 ? double(p.batch_b) : loops) * Kt * p.Nt * b_passes;
    }
    const double dram_bytes = (p.a.in_l1 ? 0.0 : a_total * a_bytes) + (p.b.in_l1 ? 0.0 : b_total * b_bytes);
    return {products * cycles_per_product, received / hw.noc_bytes_per_cycle, dram_bytes / hw.dram_bytes_per_cycle};
}

double estimated_cycles(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    return roofline(p, hw, family, b).cycles();
}

namespace {

uint32_t cores_used(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    if (family == Family::Reuse) {
        const uint32_t blocks = p.batch_a * p.Mt / b.per_core_M;
        return std::min(blocks, static_cast<uint32_t>(hw.grid.x * hw.grid.y));
    }
    return div_up(output_rows(p, b.fuse_batch), b.per_core_M) * div_up(p.Nt, b.per_core_N);
}

// Subblocks two tiles or more on each side, unless B's tiles are smaller than A's. Per K step, an h x w
// subblock (h <= w) unpacks h tiles of A and h * w of B, so going from 1 x 8 to 2 x 4 costs one more A tile
// per 8 outputs, and avoids the single-row path, whose per-tile overhead shows as up to 10% on large
// matmuls with bf16 B. Only when A is the heavier operand (e.g. bf16 A, block-float B) does the extra A
// unpack cost more than it saves (1 x 8 then wins by ~5% on the Wormhole sweeps).
void set_subblock(const Problem& p, const EmpiricalDefaults& d, Family family, Blocking& b) {
    const bool reuse = family == Family::Reuse;
    const bool prefer_two_wide =
        d.prefer_two_wide_subblocks && tt::tile_size(p.in1_format) >= tt::tile_size(p.in0_format);
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
}

}  // namespace

namespace {

// A sharded operand or output fixes the family, grid and per-core sizes; returns that layout's candidate, or
// nothing when the combination isn't supported (or doesn't fit), in which case the caller falls back.
std::vector<Candidate> sharded_candidates(const Problem& p, const HardwareDesc& hw, const EmpiricalDefaults& d) {
    std::vector<Candidate> result;
    auto add = [&](Family family,
                   std::optional<Blocking> b,
                   CoreCoord grid,
                   std::optional<CoreRange> workers,
                   bool transpose_mcast) {
        if (!b) {
            return;
        }
        set_subblock(p, d, family, *b);
        HardwareDesc sub = hw;
        sub.grid = grid;
        result.push_back({family, *b, cores_used(p, sub, family, *b), grid, workers, transpose_mcast});
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
        if (a.layout == Layout::WidthSharded) {
            // 1D in0-mcast on A's grid: each core holds all of M for a slice of K, and computes a slice of N
            if (b_batched || p.b.sharded() || a.col_major || a.shard_h != M) {
                return result;
            }
            BlockRules rules = out_rules;
            rules.k_divides = a.shard_w;
            rules.k_preferred = a.shard_w;
            const auto per_core_N = div_up(p.Nt, a.shard_cores);
            add(Family::Mcast1DIn0,
                block_1d(p, hw, d, Family::Mcast1DIn0, M, per_core_N, true, rules),
                grid,
                a.shard_grid,
                false);
        } else if (a.layout == Layout::HeightSharded) {
            // Each core holds a slice of M over all of K
            if (a.col_major || a.shard_w != p.Kt) {
                return result;
            }
            if (b_batched) {
                // Reuse on A's grid; B interleaved, or sharded exactly like A
                if (p.b.sharded() && !p.b_shard_matches_a) {
                    return result;
                }
                const bool divides = p.Mt % a.shard_h == 0;
                const bool whole_batches = a.shard_h % p.Mt == 0 && M % a.shard_h == 0;
                if (!divides && !whole_batches) {
                    return result;
                }
                add(Family::Reuse, block_reuse_sharded(p, hw), grid, std::nullopt, false);
            } else {
                // 1D in1-mcast on A's grid, reading A in place (in0_block_w = K)
                if (p.b.sharded()) {
                    return result;
                }
                BlockRules rules = out_rules;
                rules.k_fixed = p.Kt;
                add(Family::Mcast1DIn1,
                    block_1d(p, hw, d, Family::Mcast1DIn1, a.shard_h, p.Nt, true, rules),
                    grid,
                    a.shard_grid,
                    false);
            }
        } else if (a.layout == Layout::BlockSharded) {
            // 2D on A's grid: rows of cores split M, columns split K (for A) and N (for the output). On a single
            // row or column of cores one of those splits is trivial.
            if (b_batched || p.b.sharded()) {
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
                block_2d(p, hw, d, per_core_M, per_core_N, true, rules),
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
        const bool block = out.layout == Layout::BlockSharded;
        const bool row_of_cores = out.layout == Layout::WidthSharded || (block && grid.y == 1 && grid.x > 1);
        const bool col_of_cores = out.layout == Layout::HeightSharded || (block && grid.x == 1 && grid.y > 1);
        if (row_of_cores && fits(1, out.shard_cores)) {
            add(Family::Mcast1DIn0,
                block_1d(p, hw, d, Family::Mcast1DIn0, M, shard_w, true, rules),
                grid,
                out.shard_grid,
                false);
        } else if (col_of_cores && fits(out.shard_cores, 1)) {
            add(Family::Mcast1DIn1,
                block_1d(p, hw, d, Family::Mcast1DIn1, shard_h, p.Nt, true, rules),
                grid,
                out.shard_grid,
                false);
        } else if (block && fits(grid.y, grid.x)) {
            add(Family::Mcast2D, block_2d(p, hw, d, shard_h, shard_w, true, rules), grid, out.shard_grid, false);
        }
        if (result.empty()) {
            // The spec's grid doesn't match the output (e.g. one core for several batches' worth of shards).
            // matmul rebuilds the output spec from the config, so keep the shard shape and derive the grid.
            const uint32_t rows = div_up(M, shard_h);
            const uint32_t cols = div_up(p.Nt, shard_w);
            const auto workers = pinned_workers(hw);
            if (block && rows <= hw.grid.y && cols <= hw.grid.x) {
                add(Family::Mcast2D, block_2d(p, hw, d, shard_h, shard_w, true, rules), hw.grid, workers, false);
            } else if (out.layout != Layout::HeightSharded && rows == 1 && cols <= cores) {
                add(Family::Mcast1DIn0,
                    block_1d(p, hw, d, Family::Mcast1DIn0, M, shard_w, true, rules),
                    hw.grid,
                    workers,
                    false);
            } else if (out.layout != Layout::WidthSharded && cols == 1 && rows <= cores) {
                add(Family::Mcast1DIn1,
                    block_1d(p, hw, d, Family::Mcast1DIn1, shard_h, p.Nt, true, rules),
                    hw.grid,
                    workers,
                    false);
            }
        }
    } else if (out.layout == Layout::WidthSharded) {
        add(Family::Mcast1DIn0,
            block_1d(p, hw, d, Family::Mcast1DIn0, M, div_up(p.Nt, cores), true, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    } else if (out.layout == Layout::HeightSharded) {
        add(Family::Mcast1DIn1,
            block_1d(p, hw, d, Family::Mcast1DIn1, div_up(M, cores), p.Nt, true, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    } else if (out.layout == Layout::BlockSharded) {
        add(Family::Mcast2D,
            block_2d(p, hw, d, div_up(M, hw.grid.y), div_up(p.Nt, hw.grid.x), true, rules),
            hw.grid,
            pinned_workers(hw),
            false);
    }
    return result;
}

}  // namespace

std::vector<Candidate> candidates(const Problem& p, const HardwareDesc& hw, const EmpiricalDefaults& d) {
    if (p.a.sharded() || p.b.sharded() || p.out.sharded()) {
        return sharded_candidates(p, hw, d);
    }
    std::vector<Candidate> result;
    const uint32_t cores = hw.grid.x * hw.grid.y;
    // On a sub-device the configs name its cores; the factories otherwise start at (0, 0)
    const std::optional<CoreRange> workers = pinned_workers(hw);
    auto add = [&](Family family, std::optional<Blocking> b) {
        if (!b) {
            return;
        }
        set_subblock(p, d, family, *b);
        result.push_back({family, *b, cores_used(p, hw, family, *b), hw.grid, workers, false});
    };
    // The mcast kernels can't take block-float B with A tiles shorter than 16 rows; Reuse can
    const bool mcast_ok = !(is_block_float(p.in1_format) && p.in0_tile_h < 16);
    if (broadcasts_a(p)) {
        if (mcast_ok && !p.no_mcast_1d) {
            add(Family::Mcast1DIn1, block_1d(p, hw, d, Family::Mcast1DIn1, div_up(p.Mt, cores), p.Nt, false));
        }
        return result;
    }
    // Batched B can't be fused into M: the mcast families then loop over the batch. Neither can a transposed A
    // of multi-row batch matrices (its reads would cross batch boundaries).
    const bool fuse_batch = p.batch_b == 1 && !(p.transpose_a && p.batch_a > 1 && p.Mt > 1);
    // Reuse needs matching batches: batched B, or (when the mcast kernels can't run it) a single batch
    if (p.batch_a == p.batch_b && (p.batch_b > 1 || !mcast_ok)) {
        add(Family::Reuse, block_reuse(p, hw, d));
    }
    if (!mcast_ok) {
        return result;
    }
    const uint32_t M = output_rows(p, fuse_batch);
    add(Family::Mcast2D, block_2d(p, hw, d, div_up(M, hw.grid.y), div_up(p.Nt, hw.grid.x), fuse_batch));
    if (!p.no_mcast_1d) {
        add(Family::Mcast1DIn0, block_1d(p, hw, d, Family::Mcast1DIn0, M, div_up(p.Nt, cores), fuse_batch));
        add(Family::Mcast1DIn1, block_1d(p, hw, d, Family::Mcast1DIn1, div_up(M, cores), p.Nt, fuse_batch));
    }
    return result;
}

std::optional<Candidate> choose_candidate(const Problem& p, const HardwareDesc& hw, const EmpiricalDefaults& d) {
    const auto all = candidates(p, hw, d);
    if (p.a.sharded() || p.b.sharded() || p.out.sharded()) {
        // The layout already fixed the family
        return all.empty() ? std::nullopt : std::optional<Candidate>(all.front());
    }
    // The candidate with the smallest roofline estimate; ties keep the earlier family
    const Candidate* best = nullptr;
    double best_cycles = 0;
    for (const auto& c : all) {
        const double cycles = estimated_cycles(p, hw, c.family, c.blocking);
        if (!best || cycles < best_cycles) {
            best = &c;
            best_cycles = cycles;
        }
    }
    if (best) {
        return *best;
    }
    return std::nullopt;
}

std::optional<MatmulProgramConfig> select_program_config(
    const Problem& p, const HardwareDesc& hw, const EmpiricalDefaults& d) {
    if (p.Mt == 0 || p.Kt == 0 || p.Nt == 0 || hw.grid.x == 0 || hw.grid.y == 0) {
        return std::nullopt;
    }
    const auto chosen = choose_candidate(p, hw, d);
    if (!chosen) {
        return std::nullopt;
    }
    const Family family = chosen->family;
    const Blocking& b = chosen->blocking;
    const std::optional<CoreRangeSet> worker_cores =
        chosen->worker_cores ? std::optional<CoreRangeSet>(CoreRangeSet(*chosen->worker_cores)) : std::nullopt;
    switch (family) {
        case Family::Mcast2D:
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = chosen->grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .transpose_mcast = chosen->transpose_mcast,
                .fused_activation = p.activation,
                .fuse_batch = b.fuse_batch,
                .allowed_worker_cores = worker_cores,
            };
        case Family::Mcast1DIn0:
        case Family::Mcast1DIn1:
            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = chosen->grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .fuse_batch = b.fuse_batch,
                // in0 reuse (broadcast A) can't fuse the activation; matmul then applies it separately
                .fused_activation = broadcasts_a(p) && !p.a.sharded() ? std::nullopt : p.activation,
                .mcast_in0 = family == Family::Mcast1DIn0,
                .allowed_worker_cores = worker_cores,
            };
        case Family::Reuse:
            return MatmulMultiCoreReuseProgramConfig{
                .compute_with_storage_grid_size = chosen->grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .allowed_worker_cores = worker_cores,
            };
    }
    return std::nullopt;
}

namespace {

// The selector's view of a memory config (and the tensor's shard spec, when it has one), with shard sizes in
// the tensor's tiles (tile_h x tile_w). nullopt, with the reason, for placements it doesn't handle.
std::optional<Placement> placement_of(
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec,
    uint32_t tile_h,
    uint32_t tile_w,
    std::string& why) {
    using tt::tt_metal::TensorMemoryLayout;
    Placement placement;
    placement.in_l1 = memory_config.buffer_type() == tt::tt_metal::BufferType::L1;
    switch (memory_config.memory_layout()) {
        case TensorMemoryLayout::INTERLEAVED: return placement;
        case TensorMemoryLayout::HEIGHT_SHARDED: placement.layout = Layout::HeightSharded; break;
        case TensorMemoryLayout::WIDTH_SHARDED: placement.layout = Layout::WidthSharded; break;
        case TensorMemoryLayout::BLOCK_SHARDED: placement.layout = Layout::BlockSharded; break;
        default: why = "unsupported memory layout"; return std::nullopt;
    }
    if (!placement.in_l1) {
        why = "DRAM-sharded tensor";
        return std::nullopt;
    }
    if (memory_config.nd_shard_spec().has_value() && !shard_spec.has_value()) {
        why = "ND-sharded tensor";
        return std::nullopt;
    }
    if (shard_spec.has_value()) {
        const auto& spec = shard_spec.value();
        if (spec.shape[0] % tile_h != 0 || spec.shape[1] % tile_w != 0) {
            why = "shard shape not a whole number of tiles";
            return std::nullopt;
        }
        placement.has_shard_spec = true;
        placement.shard_grid = spec.grid.bounding_box();
        placement.shard_cores = spec.grid.num_cores();
        placement.shard_h = spec.shape[0] / tile_h;
        placement.shard_w = spec.shape[1] / tile_w;
        placement.col_major = spec.orientation == tt::tt_metal::ShardOrientation::COL_MAJOR;
    }
    return placement;
}

// B sharded in DRAM (width or ND): the 2D factory reads it in place; the other factories can't.
bool is_dram_sharded_b(const Tensor& b) {
    const auto& mc = b.memory_config();
    return mc.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           (mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
            (mc.nd_shard_spec().has_value() && !b.shard_spec().has_value()));
}

}  // namespace

std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported) {
    std::string why;
    auto reject = [&](std::string reason) -> std::optional<MatmulProgramConfig> {
        if (unsupported != nullptr) {
            *unsupported = std::move(reason);
        }
        return std::nullopt;
    };
    const auto& output_mem_config = attributes.output_mem_config;

    // Tiles: A's are in0_tile_h x 32, B's 32 x in1_tile_w (matmul validation requires the 32-wide K side)
    const auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
    const auto in1_tile = utilities::get_matmul_tile(input_tensor_b, transpose_b);
    const auto out_tile =
        attributes.output_tile.value_or(tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()}));
    if (in0_tile.get_width() != TILE_DIM || in1_tile.get_height() != TILE_DIM) {
        return reject("tile K side not 32");
    }

    const bool b_dram_sharded = is_dram_sharded_b(input_tensor_b);
    const auto a_placement =
        placement_of(input_tensor_a.memory_config(), input_tensor_a.shard_spec(), in0_tile.get_height(), TILE_DIM, why);
    if (!a_placement) {
        return reject("A: " + why);
    }
    std::optional<Placement> b_placement = Placement{};
    if (!b_dram_sharded) {
        b_placement = placement_of(
            input_tensor_b.memory_config(), input_tensor_b.shard_spec(), TILE_DIM, in1_tile.get_width(), why);
        if (!b_placement) {
            return reject("B: " + why);
        }
    }
    const auto out_placement = placement_of(
        output_mem_config, output_mem_config.shard_spec(), in0_tile.get_height(), in1_tile.get_width(), why);
    if (!out_placement) {
        return reject("output: " + why);
    }
    // A sharded tensor must carry its shard spec (an output's may be left to the program config)
    if ((a_placement->sharded() && !a_placement->has_shard_spec) ||
        (b_placement->sharded() && !b_placement->has_shard_spec)) {
        return reject("sharded input without a shard spec");
    }
    if (b_dram_sharded && (a_placement->sharded() || out_placement->sharded())) {
        return reject("DRAM-sharded B with a sharded A or output");
    }
    if (out_tile.get_width() != in1_tile.get_width()) {
        return reject("output tile wider than B's tile");
    }

    const auto a_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto b_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    if (a_shape.rank() < 2 || b_shape.rank() < 2) {
        return reject("rank below 2");
    }
    Problem p;
    p.batch_a = a_shape.volume() / (a_shape[-2] * a_shape[-1]);
    p.batch_b = b_shape.volume() / (b_shape[-2] * b_shape[-1]);
    p.in0_tile_h = in0_tile.get_height();
    p.in1_tile_w = in1_tile.get_width();
    p.out_tile_h = out_tile.get_height();
    p.out_tile_w = out_tile.get_width();
    p.Mt = a_shape[-2] / p.in0_tile_h;
    p.Kt = a_shape[-1] / TILE_DIM;
    p.Nt = b_shape[-1] / p.in1_tile_w;
    if (p.batch_b > 1 && p.batch_a != p.batch_b) {
        // Only a batch of one broadcasts (1D in1-mcast in0 reuse): interleaved tensors of equal rank >= 3
        if (p.batch_a != 1 || a_shape.rank() != b_shape.rank() || a_shape.rank() < 3) {
            return reject("A and B batches don't match");
        }
        if (a_placement->sharded() || b_placement->sharded() || out_placement->sharded()) {
            return reject("broadcast A with a sharded tensor");
        }
    }
    p.a = *a_placement;
    p.b = *b_placement;
    p.out = *out_placement;
    p.no_mcast_1d = attributes.global_cb.has_value() || b_dram_sharded;

    if (p.a.sharded() && p.b.sharded()) {
        const auto& sa = input_tensor_a.shard_spec().value();
        const auto& sb = input_tensor_b.shard_spec().value();
        p.b_shard_matches_a = p.a.layout == p.b.layout && sa.grid == sb.grid && sa.orientation == sb.orientation;
    }
    // A sharded tensor's shard spec is physical while transpose_a transposes the logical shape
    if (transpose_a && p.a.sharded()) {
        return reject("transpose_a with a sharded A");
    }

    if (!attributes.compute_kernel_config.has_value()) {
        return reject("no compute kernel config");
    }
    auto* device = input_tensor_a.device();
    const auto arch = device->arch();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config.value());
    p.in0_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    p.in1_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    p.out_format =
        tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value_or(input_tensor_a.dtype()));
    if (b_dram_sharded && (p.batch_b > 1 || (is_block_float(p.in1_format) && p.in0_tile_h < 16))) {
        return reject("DRAM-sharded B that only 2D could read, but 2D can't run");
    }
    p.bias_tile_bytes = bias_single_tile_size;
    p.transpose_a = transpose_a;
    p.math_fidelity = math_fidelity;
    p.fp32_dest_acc_en = fp32_dest_acc_en;
    p.packer_l1_acc = packer_l1_acc;
    p.dst_full_sync_en = dst_full_sync_en;
    // Fuse the activation only if the kernels support it; otherwise matmul applies it as a separate op
    if (attributes.user_fused_activation.has_value() &&
        utilities::is_fusable_activation(attributes.user_fused_activation->op_type)) {
        p.activation = attributes.user_fused_activation;
    }

    // Worker grid: the device's, or on a sub-device its worker rectangle (the factories anchor their grid at
    // its first core); a user core_grid shrinks it
    auto grid = device->compute_with_storage_grid_size();
    CoreCoord origin{0, 0};
    const bool on_sub_device = attributes.sub_device_id.has_value();
    if (on_sub_device) {
        const auto cores =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, attributes.sub_device_id.value());
        const auto bbox = cores.bounding_box();
        if (cores.num_cores() != bbox.size()) {
            return reject("sub-device worker cores not a rectangle");
        }
        grid = bbox.grid_size();
        origin = bbox.start_coord;
    }
    if (attributes.user_core_coord.has_value()) {
        const auto& user = attributes.user_core_coord.value();
        if (user.x > 0 && user.y > 0) {
            grid = CoreCoord(std::min(user.x, grid.x), std::min(user.y, grid.y));
        }
    }

    // L1 left for CBs: free space above the lowest L1 buffer, less this op's own L1 output (not allocated yet)
    uint32_t budget = utilities::get_max_l1_space(input_tensor_a);
    if (output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 && !p.out.sharded()) {
        const uint64_t out_tiles = static_cast<uint64_t>(std::max(p.batch_a, p.batch_b)) * p.Mt * p.Nt;
        const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
        const uint64_t out_per_bank =
            div_up(out_tiles, num_banks) * static_cast<uint64_t>(out_tile_bytes(p, p.out_format));
        budget = out_per_bank >= budget ? 0 : budget - static_cast<uint32_t>(out_per_bank);
    }
    const auto d = EmpiricalDefaults::for_arch(arch);
    budget = budget > d.l1_headroom_bytes ? budget - d.l1_headroom_bytes : 0;

    auto hw = HardwareDesc::for_arch(arch, grid, budget);
    hw.origin = origin;
    hw.pinned_origin = on_sub_device;
    if (auto config = select_program_config(p, hw, d)) {
        return config;
    }
    // Nothing blocked fits: the non-reusing factory still runs all-interleaved 32x32 inputs on the device grid
    const bool all_interleaved = !p.a.sharded() && !p.b.sharded() && !p.out.sharded() && !b_dram_sharded;
    const bool full_tiles = p.in0_tile_h == TILE_DIM && p.in1_tile_w == TILE_DIM;
    if (all_interleaved && full_tiles && !on_sub_device && !broadcasts_a(p)) {
        return MatmulMultiCoreProgramConfig{};
    }
    if (p.a.sharded() || p.b.sharded() || p.out.sharded()) {
        return reject("unsupported sharded layout combination, or it doesn't fit L1");
    }
    if (is_block_float(p.in1_format) && p.in0_tile_h < 16 && p.batch_a != p.batch_b) {
        return reject("block-float B with A tiles under 16 rows needs Reuse, which can't broadcast B over A's batch");
    }
    return reject("no config fits L1");
}

}  // namespace ttnn::operations::matmul::auto_config
