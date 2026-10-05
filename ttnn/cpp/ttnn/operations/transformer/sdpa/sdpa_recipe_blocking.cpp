// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_blocking.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "sdpa_recipe.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
namespace {

constexpr uint32_t kTile = 32;
constexpr uint32_t kBf16Tile = 2048;
constexpr uint32_t kDefaultQTiles = 8;
constexpr uint32_t kDefaultKTiles = 16;
// Prefer Q256/K512, the most-measured blocking, whenever it fits and its modeled cost is within this
// fraction of the cheapest candidate: the cost model is not more precise than that.
constexpr double kDefaultPreference = 0.05;
// Ring layouts whose preferred (double-buffered Q) layout does not fit lose Q prefetch.
constexpr double kFallbackPenalty = 1.03;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Per (Q chunk, K chunk) block cost model, a roofline in units of one D128 QK+PV tile product:
//   compute   = c * (q_tiles * k_tiles * d_tiles / 4 + ck * q_tiles * d_tiles / 4)
//   bandwidth = bw * k_tiles * d_tiles / 4
//   cost      = max(compute, bandwidth)
// `ck` is the per-row, per-K-chunk softmax state work (so short K chunks cost more per key) and
// `bw` the Q tile count below which streaming the K/V chunk into the core dominates (so short Q
// chunks cost more per row). Per variant, fitted to matched-chunk trace timings (single P150b,
// 10 heads, 8192 x 8192, D128, full grid):
// `c` from Q256/K512, `ck` from Q256/K256 (it also predicts Q320/K256 within 2%), `bw` from
// Q128/K512. STANDARD and LOW_PRECISION refit 2026-10-02 on the reference-max state kernels (one Blackhole
// Galaxy chip, same shapes; BFP4 stays compute-bound at Q128/K512, so its bw is kept).
//
// A core's makespan adds pipeline fill/drain the block roofline does not see (block_overhead): the
// first Q chunk is read before any compute and the last output chunk is written after it (per core,
// proportional to the Q chunk), and each Q chunk's first K/V block arrives before its first matmul
// (per job, proportional to that block's keys). Both are negligible at long K (16+ K blocks per Q
// chunk) and dominate short-K cross attention (one K block per Q chunk: LTX-2 text / A2V cross,
// K32/K256), where the roofline alone ties every Q chunk that divides the heads evenly over the
// cores and the tie went to the largest Q (e.g. FAST Q384/K32, 1.19x the best measured blocking).
// Fitted on dense trace timings of 7 short/medium-K DiT shapes x FAST/STANDARD/BALANCED/
// LOW_PRECISION(BFP8) over Q128-512 x K32-512 (bh-38, 1x2 mesh, 16 back-to-back ops per trace).
constexpr double kQFillDrain = 12.0;  // x c x q_tiles x d_tiles / 4, once per core
constexpr double kKFill = 0.5;        // x bw x (first K block's tiles) x d_tiles / 4, once per Q chunk
struct BlockCostModel {
    double c;
    double ck;
    double bw;
};

BlockCostModel block_cost_model(const PrecisionPolicy& policy) {
    switch (policy.selection.recipe) {
        case Recipe::A: return {1.000, 1.49, 5.99};
        // Fused STANDARD chunks (refit 2026-10-05, dense 10 x 8192^2 D128 on one Blackhole chip).
        case Recipe::B: return {0.984, 2.20, 6.15};
        case Recipe::C: return {1.878, 3.58, 9.46};
        case Recipe::D: return {2.436, 3.14, 11.75};
        case Recipe::E:
            switch (policy.selection.kv_storage) {
                // Fused LOW_PRECISION chunks (fitted 2026-10-04, dense 10 x 8192^2 D128 on one Blackhole chip).
                case KVStorage::BF16: return {0.722, 2.83, 6.09};
                case KVStorage::BFP8: return {0.648, 3.57, 3.86};
                case KVStorage::BFP4: return {0.648, 3.62, 3.41};
            }
    }
    TT_THROW("Unknown SDPA precision recipe");
}

// Build-flag facts of a geometry, read from the recipe program the factory would build.
struct RecipeBuild {
    uint64_t cb_bytes = 0;
    uint64_t fused_cb_bytes = 0;  // CBs 29-31 of fused chunks, which a factory drops when they do not fit
    uint32_t qk_width = 4;        // SDPA_RECIPE_QK_W
    uint32_t pv_width = 4;        // SDPA_RECIPE_PV_W
};

// Compute slowdown not in the fitted model: narrower matmul subblocks.
double subblock_penalty(uint32_t width) { return width >= 4 ? 1.0 : width == 2 ? 1.10 : 1.30; }
// LOW_PRECISION's fused chunks pay their per-subblock pack-thread work (exp, P and row-sum packs) per QK subblock:
// a width-2 QK subblock is 1.18x slower overall (1 core, BFP8: Q192/K960 2.55 vs K768 2.96, K1024 3.03 TF), and
// width 1 runs the unfused kernel.
double qk_subblock_penalty(const PrecisionPolicy& policy, uint32_t width) {
    if (policy.selection.recipe == Recipe::E) {
        return width >= 4 ? 1.0 : width == 2 ? 1.36 : 1.60;
    }
    return subblock_penalty(width);
}

double block_cost(
    const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    const RecipeBuild& build,
    bool dense) {
    const auto m = block_cost_model(policy);
    const double d = d_tiles / 4.0;
    // The q*k*d term is the QK and PV matmuls in equal parts; each pays its own subblock-width penalty
    // (a one-tile K chunk narrows QK only, so it does not slow PV or the softmax state work).
    const double matmul = 0.5 * (qk_subblock_penalty(policy, build.qk_width) + subblock_penalty(build.pv_width));
    // Dense/joint STANDARD computes an odd Q chunk with one padding row (recipe_compute_q_tiles).
    const uint32_t rows = dense ? recipe_compute_q_tiles(policy, q_tiles) : q_tiles;
    const double compute = m.c * (static_cast<double>(rows) * k_tiles * d * matmul + m.ck * rows * d);
    return std::max(compute, m.bw * k_tiles * d);
}

// Pipeline fill/drain of one core running `jobs` Q chunks (see kQFillDrain / kKFill). `first_k_tiles`
// is the K tiles of a Q chunk's first K/V block (the chunk, or the whole sequence when shorter).
double block_overhead(const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t first_k_tiles, uint32_t d_tiles, uint32_t jobs) {
    const auto m = block_cost_model(policy);
    const double d = d_tiles / 4.0;
    return kQFillDrain * m.c * q_tiles * d + kKFill * jobs * m.bw * first_k_tiles * d;
}

uint64_t cb_bytes(const tt::tt_metal::ProgramDescriptor& program) {
    uint64_t bytes = 0;
    for (const auto& cb : program.cbs) {
        bytes += cb.total_size;
    }
    return bytes;
}

// The recipe's own circular buffers and build flags (recipe_compute_program), which dense, joint
// and the B-E ring and exp-ring program factories adopt as their compute layout.
RecipeBuild recipe_build(const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
    static std::mutex mutex;
    static std::map<std::tuple<uint8_t, uint8_t, uint32_t, uint32_t, uint32_t>, RecipeBuild> memo;
    const auto key = std::make_tuple(
        static_cast<uint8_t>(policy.selection.recipe),
        static_cast<uint8_t>(policy.selection.kv_storage),
        q_tiles,
        k_tiles,
        d_tiles);
    {
        std::lock_guard lock(mutex);
        if (auto it = memo.find(key); it != memo.end()) {
            return it->second;
        }
    }
    const CoreRangeSet core(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    const auto program = recipe_compute_program(policy, core, 1, q_tiles, k_tiles, d_tiles);
    RecipeBuild build{.cb_bytes = cb_bytes(program)};
    for (const auto& cb : program.cbs) {
        const uint8_t index = cb.format_descriptors.front().buffer_index;
        build.fused_cb_bytes += (index == 29 || index == 30 || index == 31) ? cb.total_size : 0;
    }
    for (const auto& [name, value] : program.kernels.front().defines) {
        if (name == "SDPA_RECIPE_QK_W") {
            build.qk_width = std::stoul(value);
        } else if (name == "SDPA_RECIPE_PV_W") {
            build.pv_width = std::stoul(value);
        }
    }
    std::lock_guard lock(mutex);
    if (memo.size() > 65536) {
        memo.clear();
    }
    memo.emplace(key, build);
    return build;
}

// Ring joint (ring_joint_sdpa_program_factory.cpp) adds to the B-E recipe layout: the lightweight
// mask (neginf plus up to two partial-tile masks), the scale scalar, the two 4 KiB state-transfer
// pages (CBs 17/18), the signal page and the derived-geometry mailbox. Partial masks are counted
// unconditionally, so this can only over-estimate.
constexpr uint64_t kRingRecipeExtraBytes = 3 * kBf16Tile + kBf16Tile + 2 * 4096 + 16 + 64;

// FAST (A) on ring keeps the legacy streaming ring layout (named_compute == false in
// ring_joint_sdpa_program_factory.cpp), all BF16 tiles: Q x q_buffer_factor, K/V double buffered,
// mask (<= 3), scale/identity/column identity, QK, out_im A/B, max/sum A/B, exp_max_diff, cb_out
// (counted unshrunk), stats_in, prev_out, stats_out, reciprocal scratch, sum_out/sum_in, plus the
// signal page and derived-geometry mailbox. Mirrors that factory while FAST keeps the legacy ring kernels.
uint64_t legacy_ring_fast_bytes(uint32_t q, uint32_t k, uint32_t d, uint32_t q_buffer_factor) {
    const uint64_t tiles = uint64_t{q} * d * q_buffer_factor + 4ull * k * d + 3 + 3 + uint64_t{q} * k +
                           2ull * q * d + 4ull * q + q + uint64_t{q} * d + q + uint64_t{q} * d + q + 1 + 2ull * q;
    return tiles * kBf16Tile + 16 + 64;
}

// FAST (A) on exp ring keeps the legacy exp-ring layout (named_compute == false in
// exp_ring_joint_sdpa_program_factory.cpp): per-pass resident Q, or one streamed Q chunk when the
// resident layout does not fit. Same table as MiniMax H3's `_exp_sdpa_l1_bytes`, which reproduces
// the factory's 1,302,528 B at Q224/K512.
uint64_t legacy_exp_ring_fast_bytes(uint32_t q, uint32_t k, uint32_t d, uint32_t passes, bool resident_q) {
    const uint64_t tiles = uint64_t{resident_q ? passes : 1u} * q * d + 4ull * k * d + 7 + 2ull * passes * q +
                           uint64_t{passes} * q * d + q + 16 + uint64_t{q} * k + 2ull * q * d + 4ull * q + q;
    return tiles * kBf16Tile;
}

// Legacy exp-ring FAST needs its streaming compute path (the kernel static_asserts on it); mirrors
// `use_streaming_compute` in exp_ring_joint_sdpa_program_factory.cpp at BF16 destination (8 tiles).
bool legacy_exp_ring_streaming(uint32_t q, uint32_t k) {
    constexpr uint32_t dst = 8;
    constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks{{{2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7},
                                                                      {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
                                                                      {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1},
                                                                      {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1}}};
    for (const auto& [h, w] : subblocks) {
        if (h * w <= dst && q % h == 0 && k % w == 0) {
            return h <= 2 && k % (dst / h) == 0 && q / h > 1;
        }
    }
    return false;
}

using ProblemKey = std::tuple<
    uint8_t,
    uint8_t,
    uint8_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    std::size_t,
    std::size_t,
    uint32_t,
    uint64_t,
    uint32_t,
    uint32_t,
    uint32_t,
    bool>;

ProblemKey key_of(const RecipeBlockingProblem& p) {
    return {
        static_cast<uint8_t>(p.op),
        static_cast<uint8_t>(p.policy.selection.recipe),
        static_cast<uint8_t>(p.policy.selection.kv_storage),
        p.batch,
        p.q_heads,
        p.q_rows,
        p.joint_q_rows,
        p.k_rows,
        p.joint_k_rows,
        p.ring_size,
        p.d_tiles,
        p.grid.x,
        p.grid.y,
        p.max_cores_per_head_batch,
        p.l1_bytes,
        p.mask_page_bytes,
        p.fixed_q_tiles,
        p.fixed_k_tiles,
        p.exp_mux_on_bottom_row};
}

std::vector<uint32_t> tile_range(uint32_t fixed, uint32_t lo, uint32_t hi) {
    if (fixed != 0) {
        return {fixed};
    }
    std::vector<uint32_t> values;
    for (uint32_t t = hi; t >= lo; --t) {
        values.push_back(t);
    }
    return values;
}

}  // namespace

std::optional<std::string> recipe_geometry_rejection(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
    // The only statement of supported recipe geometry: the chooser consults it, and the ring / exp-ring
    // entry points and device operations validate through validate_recipe_geometry below. Dense/joint
    // also mirror it in recipe_dense_q_tiles / recipe_dense_k_tiles (sdpa_recipe.cpp).
    if (q_tiles == 0 || k_tiles == 0 || d_tiles == 0) {
        return std::string("recipes require tile-aligned, nonzero Q/K chunks and head dims");
    }
    // B-E on every op: any tile-aligned Q chunk up to the recurrent-state arrays (32 tile rows), any
    // tile-aligned K chunk and head dim; L1 fit is recipe_l1_bytes.
    if (q_tiles > 32) {
        return fmt::format("Q chunk {} exceeds 1024 rows", q_tiles * kTile);
    }
    if (op == RecipeOp::Dense || op == RecipeOp::Joint || policy.selection.recipe != Recipe::A) {
        return std::nullopt;
    }
    // FAST ring / exp ring keep the legacy ring kernels, which support only these geometries.
    if (d_tiles != 2 && d_tiles != 4 && d_tiles != 8) {
        return fmt::format("head dim {} is not 64, 128 or 256", d_tiles * kTile);
    }
    if (q_tiles < 4 || q_tiles > 10) {
        return fmt::format("Q chunk {} is outside 128-320 rows", q_tiles * kTile);
    }
    if (k_tiles != 8 && k_tiles != 12 && k_tiles != 16) {
        return fmt::format("K chunk {} is not 256, 384 or 512 rows", k_tiles * kTile);
    }
    if (op == RecipeOp::ExpRing && (k_tiles != 16 || d_tiles != 4)) {
        return std::string("exp ring recipes require K512/D128");
    }
    return std::nullopt;
}

void validate_recipe_geometry(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim) {
    const char* name = op == RecipeOp::Ring      ? "ring"
                       : op == RecipeOp::ExpRing ? "exp ring"
                       : op == RecipeOp::Joint   ? "joint"
                                                 : "dense";
    TT_FATAL(
        q_chunk_size % kTile == 0 && k_chunk_size % kTile == 0 && head_dim % kTile == 0,
        "Named {} SDPA recipes require tile-aligned Q/K chunks and head dims, got Q{}/K{}/D{}",
        name,
        q_chunk_size,
        k_chunk_size,
        head_dim);
    const auto rejection =
        recipe_geometry_rejection(op, policy, q_chunk_size / kTile, k_chunk_size / kTile, head_dim / kTile);
    TT_FATAL(
        !rejection,
        "Named {} SDPA recipes do not support Q{}/K{}/D{}: {}",
        name,
        q_chunk_size,
        k_chunk_size,
        head_dim,
        rejection.value_or(""));
}

uint32_t recipe_mask_group_rows(const PrecisionPolicy& policy, uint32_t q_tiles) {
    return policy.fp32_destination                    ? 1
           : policy.selection.recipe == Recipe::A ? (q_tiles % 2 == 0 ? 2 : 1)
                                                  : 2;
}

RecipeL1Estimate recipe_l1_bytes(
    RecipeOp op,
    const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    const RecipeL1Context& context) {
    const auto rejection = recipe_geometry_rejection(op, policy, q_tiles, k_tiles, d_tiles);
    TT_FATAL(!rejection, "Unsupported SDPA recipe geometry: {}", *rejection);
    const bool fast = policy.selection.recipe == Recipe::A;
    const uint64_t q_slot = uint64_t{q_tiles} * d_tiles * kBf16Tile;
    switch (op) {
        case RecipeOp::Dense:
        case RecipeOp::Joint: {
            const auto build = recipe_build(policy, recipe_compute_q_tiles(policy, q_tiles), k_tiles, d_tiles);
            // attn_mask CB: two row groups when they fit, one otherwise (sdpa_recipe.cpp). The minimum layout
            // also drops the fused chunks' CBs (check_recipe_l1_fit).
            const uint64_t mask_group =
                uint64_t{recipe_mask_group_rows(policy, q_tiles)} * k_tiles * context.mask_page_bytes;
            return {build.cb_bytes + 2 * mask_group, build.cb_bytes - build.fused_cb_bytes + mask_group};
        }
        case RecipeOp::Ring: {
            if (fast) {
                const uint64_t bytes =
                    legacy_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.q_blocks_per_worker > 1 ? 2 : 1);
                return {bytes, bytes};
            }
            const auto build = recipe_build(policy, q_tiles, k_tiles, d_tiles);
            const uint64_t bytes = build.cb_bytes + kRingRecipeExtraBytes;
            // The factory's fallbacks: drop the fused chunks' CBs, then single-slot Q.
            return {bytes, bytes - build.fused_cb_bytes - q_slot};
        }
        case RecipeOp::ExpRing: {
            if (fast) {
                return {
                    legacy_exp_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.passes, true),
                    legacy_exp_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.passes, false)};
            }
            // The exp-ring factory keeps the recipe layout with a single Q slot, plus the 64 B live-length
            // mailbox of a device-tensor logical_n (counted unconditionally).
            const auto build = recipe_build(policy, q_tiles, k_tiles, d_tiles);
            const uint64_t bytes = build.cb_bytes - q_slot + 64;
            return {bytes, bytes - build.fused_cb_bytes};  // the factory drops the fused chunks' CBs to fit
        }
    }
    TT_THROW("Unknown SDPA recipe op");
}

std::vector<RecipeBlocking> recipe_blocking_candidates(const RecipeBlockingProblem& p) {
    std::vector<RecipeBlocking> candidates;
    const uint32_t batch_heads = p.batch * p.q_heads;
    if (batch_heads == 0 || p.q_rows == 0 || p.k_rows == 0 || p.grid.x == 0 || p.grid.y == 0) {
        return candidates;
    }
    // A chunk longer than the sequence only adds padding; the default sizes stay in range.
    const bool dense = p.op == RecipeOp::Dense || p.op == RecipeOp::Joint;
    const uint32_t q_cap = std::min(kRecipeSearchMaxQTiles, std::max(div_up(p.q_rows + p.joint_q_rows, kTile), 10u));
    // Padded K blocks are costed, so short K picks short chunks. Fused LOW_PRECISION amortizes its per-chunk
    // work (saturation check, fold, PV pieces) over longer K chunks: up to K1024 when L1 allows.
    const uint32_t k_cap = p.policy.selection.recipe == Recipe::E ? 2 * kRecipeSearchMaxKTiles : kRecipeSearchMaxKTiles;
    const uint32_t q_floor = std::min(kRecipeSearchMinQTiles, div_up(p.q_rows + p.joint_q_rows, kTile));
    const uint32_t k_floor = std::min(kRecipeSearchMinKTiles, div_up(p.k_rows + p.joint_k_rows, kTile));
    const auto q_range = tile_range(p.fixed_q_tiles, q_floor, q_cap);
    const auto k_range = tile_range(p.fixed_k_tiles, k_floor, k_cap);
    for (auto qi = q_range.rbegin(); qi != q_range.rend(); ++qi) {  // ascending Q
        const uint32_t qt = *qi;
        bool any_fit = false;
        for (auto ki = k_range.rbegin(); ki != k_range.rend(); ++ki) {  // ascending K
            const uint32_t kt = *ki;
            if (!recipe_geometry_supported(p.op, p.policy, qt, kt, p.d_tiles)) {
                continue;
            }
            // Ring / exp ring round STANDARD's odd Q chunk up to the next even one (sdpa.cpp); the even
            // candidate is costed on its own.
            if (!dense && qt % 2 != 0 && recipe_compute_q_tiles(p.policy, qt) != qt) {
                continue;
            }
            // Dense/joint L1 grows with Q and K: stop at the first K that does not fit.
            if (dense &&
                recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, {.mask_page_bytes = p.mask_page_bytes}).minimum >
                    p.l1_bytes) {
                break;
            }
            any_fit = true;
            const uint32_t q_chunk = qt * kTile;
            const uint32_t k_chunk = kt * kTile;
            // FAST ring / exp ring keep their legacy compute; everything else builds the recipe program.
            const bool recipe_compute = dense || p.policy.selection.recipe != Recipe::A;
            const RecipeBuild build = recipe_compute ? recipe_build(p.policy, qt, kt, p.d_tiles) : RecipeBuild{};
            const double block = block_cost(p.policy, qt, kt, p.d_tiles, build, dense);
            // K tiles of a Q chunk's first K/V block: the chunk, or the whole (primary) sequence when shorter.
            const uint32_t first_k_tiles = std::min(kt, std::max(1u, div_up(p.k_rows, kTile)));
            auto admit = [&](uint32_t jobs, uint32_t k_blocks, CoreCoord grid, const RecipeL1Context& context) {
                const auto l1 = recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, context);
                if (l1.minimum > p.l1_bytes) {
                    return;
                }
                double cost = static_cast<double>(jobs) * k_blocks * block +
                              block_overhead(p.policy, qt, first_k_tiles, p.d_tiles, jobs);
                if (l1.preferred > p.l1_bytes) {
                    cost *= kFallbackPenalty;
                }
                candidates.push_back({q_chunk, k_chunk, grid, cost, jobs, l1});
            };
            switch (p.op) {
                case RecipeOp::Dense:
                case RecipeOp::Joint: {
                    // run_recipe_segments: one KV-forwarding chain per batch/head, jobs split evenly.
                    const uint32_t cores = p.grid.x * p.grid.y;
                    const uint32_t jobs_per_head = div_up(p.q_rows + p.joint_q_rows, q_chunk);
                    const uint32_t chain =
                        std::min({jobs_per_head, cores / batch_heads, p.max_cores_per_head_batch});
                    if (chain == 0) {
                        break;
                    }
                    admit(
                        div_up(jobs_per_head, chain),
                        div_up(p.k_rows + p.joint_k_rows, k_chunk),
                        p.grid,
                        RecipeL1Context{.mask_page_bytes = p.mask_page_bytes});
                    break;
                }
                case RecipeOp::Ring: {
                    // ring_joint_sdpa_program_factory.cpp: all Q chunks of all heads split evenly over
                    // the worker grid; every ring iteration streams the local KV shard, the joint KV once.
                    const uint32_t cores = p.grid.x * p.grid.y;
                    const uint32_t q_chunks = div_up(p.q_rows, q_chunk) + div_up(p.joint_q_rows, q_chunk);
                    const uint32_t jobs = div_up(batch_heads * q_chunks, cores);
                    const uint32_t k_blocks = p.ring_size * div_up(p.k_rows, k_chunk) + div_up(p.joint_k_rows, k_chunk);
                    admit(jobs, k_blocks, p.grid, RecipeL1Context{.q_blocks_per_worker = jobs});
                    break;
                }
                case RecipeOp::ExpRing: {
                    // exp_ring_joint_sdpa_device_operation.cpp: a head's Q chunks fill one core row
                    // exactly (num_q_chunks == cols * segs_per_head); rows walk head-segments as serial
                    // passes (at most 3). The joint Q length must divide by the Q chunk.
                    if (p.joint_q_rows % q_chunk != 0) {
                        break;
                    }
                    if (p.grid.x < 2 || (p.exp_mux_on_bottom_row && p.grid.y < 3)) {
                        break;
                    }
                    const uint32_t max_cols = p.exp_mux_on_bottom_row ? p.grid.x : p.grid.x - 1;
                    const uint32_t rows = p.exp_mux_on_bottom_row ? p.grid.y - 2 : p.grid.y;
                    const uint32_t q_chunks = div_up(p.q_rows, q_chunk) + p.joint_q_rows / q_chunk;
                    const uint32_t k_blocks = p.ring_size * div_up(p.k_rows, k_chunk) + div_up(p.joint_k_rows, k_chunk);
                    // An explicit Q chunk keeps the caller's grid; otherwise the op may narrow it, down
                    // to the factory's minimum of 3 SDPA columns (one pure SDPA plus two MUX writers).
                    const uint32_t min_cols = p.fixed_q_tiles != 0 ? max_cols : 3;
                    for (uint32_t cols = max_cols; cols >= std::max(min_cols, 1u); --cols) {
                        if (q_chunks % cols != 0) {
                            continue;
                        }
                        const uint32_t segs = q_chunks / cols;
                        const uint32_t segments = batch_heads * segs;
                        if (segments < rows) {
                            continue;
                        }
                        const uint32_t passes = div_up(segments, rows);
                        if (passes > 3) {
                            continue;
                        }
                        if (p.policy.selection.recipe == Recipe::A && !legacy_exp_ring_streaming(qt, kt)) {
                            continue;
                        }
                        const CoreCoord grid(p.exp_mux_on_bottom_row ? cols : cols + 1, p.grid.y);
                        admit(passes, k_blocks, grid, RecipeL1Context{.passes = passes});
                        if (cols == 1) {
                            break;
                        }
                    }
                    break;
                }
            }
        }
        if (dense && !any_fit && p.fixed_q_tiles == 0) {
            break;  // nothing fits at this Q, so no larger Q fits either
        }
    }
    // Cheapest first; ties prefer larger K, then less Q padding, then larger Q, then a wider grid.
    const uint32_t q_total = p.q_rows + p.joint_q_rows;
    auto rank = [q_total](const RecipeBlocking& c) {
        const int64_t q = c.q_chunk_size;
        const int64_t padding = static_cast<int64_t>(div_up(q_total, c.q_chunk_size)) * q - q_total;
        return std::make_tuple(c.cost, -static_cast<int64_t>(c.k_chunk_size), padding, -q, -static_cast<int64_t>(c.grid.x));
    };
    std::stable_sort(candidates.begin(), candidates.end(), [&](const RecipeBlocking& a, const RecipeBlocking& b) {
        return rank(a) < rank(b);
    });
    return candidates;
}

std::optional<RecipeBlocking> choose_recipe_blocking(const RecipeBlockingProblem& problem) {
    static std::mutex mutex;
    static std::map<ProblemKey, std::optional<RecipeBlocking>> memo;
    const auto key = key_of(problem);
    {
        std::lock_guard lock(mutex);
        if (auto it = memo.find(key); it != memo.end()) {
            return it->second;
        }
    }
    const auto candidates = recipe_blocking_candidates(problem);
    std::optional<RecipeBlocking> choice;
    if (!candidates.empty()) {
        choice = candidates.front();
        for (const auto& candidate : candidates) {
            if (candidate.cost > choice->cost * (1.0 + kDefaultPreference)) {
                break;
            }
            if (candidate.q_chunk_size == kDefaultQTiles * kTile && candidate.k_chunk_size == kDefaultKTiles * kTile) {
                choice = candidate;
                break;
            }
        }
    }
    std::lock_guard lock(mutex);
    if (memo.size() > 4096) {
        memo.clear();
    }
    memo.emplace(key, choice);
    return choice;
}

bool recipe_blocking_requested(const std::optional<SDPAProgramConfig>& program_config) {
    return !program_config || program_config->q_chunk_size == 0 || program_config->k_chunk_size == 0;
}

void reject_auto_blocking_without_recipe(const std::optional<SDPAProgramConfig>& program_config) {
    // Same wording as the legacy device-op validation: without a recipe, 0 is just an invalid chunk size.
    TT_FATAL(
        !program_config || program_config->q_chunk_size != 0,
        "q_chunk_size must be a positive multiple of TILE_SIZE. Got q_chunk_size: 0 (0 selects op-chosen blocking, "
        "which requires an explicit precision recipe)");
    TT_FATAL(
        !program_config || program_config->k_chunk_size != 0,
        "k_chunk_size must be a positive multiple of TILE_SIZE. Got k_chunk_size: 0 (0 selects op-chosen blocking, "
        "which requires an explicit precision recipe)");
}

namespace {

uint32_t fixed_tiles(std::size_t chunk) { return chunk % kTile == 0 ? static_cast<uint32_t>(chunk / kTile) : 0; }

SDPAProgramConfig apply_choice(
    SDPAProgramConfig config,
    const std::optional<RecipeBlocking>& choice,
    const RecipeBlockingProblem& problem,
    const char* op_name) {
    const bool auto_q = config.q_chunk_size == 0;
    const bool auto_k = config.k_chunk_size == 0;
    if (choice) {
        config.q_chunk_size = choice->q_chunk_size;
        config.k_chunk_size = choice->k_chunk_size;
        config.compute_with_storage_grid_size = choice->grid;
        log_debug(
            tt::LogOp,
            "SDPA recipe {} blocking: Q{}/K{} grid {}x{} (auto Q {}, auto K {}, cost {:.0f}, L1 {} B of {} B)",
            op_name,
            choice->q_chunk_size,
            choice->k_chunk_size,
            choice->grid.x,
            choice->grid.y,
            auto_q,
            auto_k,
            choice->cost,
            choice->l1.preferred,
            problem.l1_bytes);
    } else {
        // Nothing fits: fall back to the default blocking so the op's validation explains why.
        config.q_chunk_size = auto_q ? kDefaultQTiles * kTile : config.q_chunk_size;
        config.k_chunk_size = auto_k ? kDefaultKTiles * kTile : config.k_chunk_size;
    }
    return config;
}

RecipeBlockingProblem base_problem(
    RecipeOp op, const PrecisionPolicy& policy, const Tensor& q, const SDPAProgramConfig& config) {
    RecipeBlockingProblem problem;
    problem.op = op;
    problem.policy = policy;
    problem.batch = q.logical_shape()[0];
    problem.q_heads = q.logical_shape()[1];
    problem.d_tiles = div_up(q.logical_shape()[3], kTile);
    problem.grid = config.compute_with_storage_grid_size;
    problem.max_cores_per_head_batch = config.max_cores_per_head_batch;
    problem.fixed_q_tiles = fixed_tiles(config.q_chunk_size);
    problem.fixed_k_tiles = fixed_tiles(config.k_chunk_size);
    return problem;
}

bool invalid_fixed(const SDPAProgramConfig& config) {
    return (config.q_chunk_size != 0 && config.q_chunk_size % kTile != 0) ||
           (config.k_chunk_size != 0 && config.k_chunk_size % kTile != 0);
}

}  // namespace

// CB budget below the lowest live L1 buffer: global semaphores and persistent buffers occupy the top of L1 in
// a pipeline, and static CBs must not overlap them (the ring and exp ring factories check the same bound).
static uint64_t free_l1_below_live_buffers(tt::tt_metal::distributed::MeshDevice& device) {
    const auto lowest = device.lowest_occupied_compute_l1_address();
    const uint64_t top = lowest.has_value() ? static_cast<uint64_t>(*lowest) : device.l1_size_per_core();
    return top - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

std::optional<SDPAProgramConfig> resolve_dense_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const Tensor* joint_q,
    const Tensor* joint_k,
    const std::optional<SDPAProgramConfig>& program_config,
    const Tensor* attn_mask) {
    if (!recipe_blocking_requested(program_config) || q.storage_type() != StorageType::DEVICE) {
        return program_config;
    }
    auto* device = q.device();
    SDPAProgramConfig config = program_config.value_or(SDPAProgramConfig{
        .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
        .sub_core_grids = std::nullopt,
        .q_chunk_size = 0,
        .k_chunk_size = 0,
        .exp_approx_mode = std::nullopt});
    auto problem = base_problem(joint_q ? RecipeOp::Joint : RecipeOp::Dense, policy, q, config);
    problem.q_rows = q.padded_shape()[2];
    problem.k_rows = k.padded_shape()[2];
    problem.joint_q_rows = joint_q ? joint_q->padded_shape()[2] : 0;
    problem.joint_k_rows = joint_k ? joint_k->padded_shape()[2] : 0;
    problem.l1_bytes = free_l1_below_live_buffers(*device);
    problem.mask_page_bytes = attn_mask ? attn_mask->buffer()->page_size() : 0;
    const auto choice = invalid_fixed(config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(config, choice, problem, joint_q ? "joint" : "dense");
}

SDPAProgramConfig resolve_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const std::optional<Tensor>& joint_q,
    const std::optional<Tensor>& joint_k,
    uint32_t ring_size,
    const SDPAProgramConfig& program_config) {
    if (!recipe_blocking_requested(program_config)) {
        return program_config;
    }
    auto problem = base_problem(RecipeOp::Ring, policy, q, program_config);
    problem.q_rows = q.padded_shape()[2];
    problem.k_rows = k.padded_shape()[2];
    const bool has_joint = joint_q && joint_q->logical_shape()[2] > 0;
    problem.joint_q_rows = has_joint ? joint_q->padded_shape()[2] : 0;
    problem.joint_k_rows = has_joint && joint_k ? joint_k->padded_shape()[2] : 0;
    problem.ring_size = ring_size;
    problem.l1_bytes = free_l1_below_live_buffers(*q.device());
    const auto choice = invalid_fixed(program_config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(program_config, choice, problem, "ring");
}

SDPAProgramConfig resolve_exp_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const std::optional<Tensor>& joint_q,
    uint32_t ring_size,
    const SDPAProgramConfig& program_config) {
    if (!recipe_blocking_requested(program_config)) {
        return program_config;
    }
    auto problem = base_problem(RecipeOp::ExpRing, policy, q, program_config);
    problem.q_rows = q.logical_shape()[2];
    problem.k_rows = k.padded_shape()[2];
    const bool has_joint = joint_q && joint_q->logical_shape()[2] > 0;
    problem.joint_q_rows = has_joint ? joint_q->logical_shape()[2] : 0;
    problem.joint_k_rows = problem.joint_q_rows;
    problem.ring_size = ring_size;
    problem.exp_mux_on_bottom_row = ttnn::prim::exp_sdpa_mux_on_bottom_row();
    problem.l1_bytes = free_l1_below_live_buffers(*q.device());
    const auto choice = invalid_fixed(program_config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(program_config, choice, problem, "exp ring");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
