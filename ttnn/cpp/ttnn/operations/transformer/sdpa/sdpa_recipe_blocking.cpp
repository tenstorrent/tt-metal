// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_blocking.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

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
// Q128/K512. STANDARD and FAST refit 2026-10-02 on the reference-max state kernels (one Blackhole
// Galaxy chip, same shapes; BFP4 stays compute-bound at Q128/K512, so its bw is kept).
//
// A core's makespan adds pipeline fill/drain the block roofline does not see (block_overhead): the
// first Q chunk is read before any compute and the last output chunk is written after it (per core,
// proportional to the Q chunk), and each Q chunk's first K/V block arrives before its first matmul
// (per job, proportional to that block's keys). Both are negligible at long K (16+ K blocks per Q
// chunk) and dominate short-K cross attention (one K block per Q chunk: LTX-2 text / A2V cross,
// K32/K256), where the roofline alone ties every Q chunk that divides the heads evenly over the
// cores and the tie went to the largest Q (e.g. Q384/K32 with legacy numerics, 1.19x the best measured blocking).
// Fitted on dense trace timings of 7 short/medium-K DiT shapes x the recipes, FAST with BFP8 K/V, over Q128-512 x K32-512 (bh-38, 1x2 mesh, 16 back-to-back ops per trace).
constexpr double kQFillDrain = 12.0;  // x c x q_tiles x d_tiles / 4, once per core
constexpr double kKFill = 0.5;        // x bw x (first K block's tiles) x d_tiles / 4, once per Q chunk
// With more batch/heads than cores there are no K/V-forwarding chains: every Q chunk reads its head's K/V from DRAM,
// so the call's K/V traffic grows with the Q chunks per head and the block roofline (one core's stream) misses it.
// Each core's share of all jobs' K/V blocks costs kKvReread x bw x k_tiles x d_tiles / 4. Fitted on one P150b
// (ACCURATE, D64-D128, traced): bge_m3 B8 16 heads S512 with a mask is 0.455 ms at Q256/K512, 0.488 ms at Q128/K512
// (the roofline alone picked Q128); B16 and 4 x 32 heads D128 S1024 stay at Q256/K512 (0.758, 1.786 ms; Q128 0.893,
// 2.388 ms).
constexpr double kKvReread = 0.5;
struct BlockCostModel {
    double c;
    double ck;
    double bw;
};

BlockCostModel block_cost_model(const PrecisionPolicy& policy) {
    switch (policy.selection.recipe) {
        // Fused STANDARD chunks (refit 2026-10-05, dense 10 x 8192^2 D128 on one Blackhole chip).
        case Recipe::B: return {0.984, 2.20, 6.15};
        case Recipe::C: return {1.878, 3.58, 9.46};
        case Recipe::D: return {2.436, 3.14, 11.75};
        case Recipe::E:
            switch (policy.selection.kv_storage) {
                // Fused FAST chunks (fitted 2026-10-04, dense 10 x 8192^2 D128 on one Blackhole chip).
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
// FAST's fused chunks pay their per-subblock pack-thread work (exp, P and row-sum packs) per QK subblock:
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
    bool dense,
    bool keyed = false) {
    const auto m = block_cost_model(policy);
    const double d = d_tiles / 4.0;
    // The q*k*d term is the QK and PV matmuls in equal parts; each pays its own subblock-width penalty
    // (a one-tile K chunk narrows QK only, so it does not slow PV or the softmax state work).
    const double matmul = 0.5 * (qk_subblock_penalty(policy, build.qk_width) + subblock_penalty(build.pv_width));
    // Dense/joint unfused STANDARD, and paired recipes with a key range, compute an odd Q chunk with one padding row
    // (recipe_compute_q_tiles).
    const uint32_t rows = dense ? recipe_compute_q_tiles(policy, q_tiles, k_tiles, keyed, keyed) : q_tiles;
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
RecipeBuild recipe_build(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles, uint32_t vd_tiles = 0) {
    static std::mutex mutex;
    static std::map<std::tuple<uint8_t, uint8_t, uint32_t, uint32_t, uint32_t, uint32_t>, RecipeBuild> memo;
    const auto key = std::make_tuple(
        static_cast<uint8_t>(policy.selection.recipe),
        static_cast<uint8_t>(policy.selection.kv_storage),
        q_tiles,
        k_tiles,
        d_tiles,
        vd_tiles);
    {
        std::lock_guard lock(mutex);
        if (auto it = memo.find(key); it != memo.end()) {
            return it->second;
        }
    }
    const CoreRangeSet core(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    const auto program = recipe_compute_program(policy, core, 1, q_tiles, k_tiles, d_tiles, std::nullopt, vd_tiles);
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
    uint32_t,
    std::size_t,
    std::size_t,
    uint32_t,
    uint64_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    bool,
    bool,
    bool,
    uint32_t,
    uint32_t,
    bool,
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
        p.vd_tiles,
        p.grid.x,
        p.grid.y,
        p.max_cores_per_head_batch,
        p.l1_bytes,
        p.mask_page_bytes,
        p.extra_l1_bytes,
        p.fixed_q_tiles,
        p.fixed_k_tiles,
        p.exp_mux_on_bottom_row,
        p.key_range,
        p.causal,
        p.sliding_window,
        p.q_offset,
        p.attention_sink,
        p.k_rows_unaligned};
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
    [[maybe_unused]] RecipeOp op,
    [[maybe_unused]] const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles) {
    // The only statement of supported recipe geometry: the chooser consults it, and the ring / exp-ring
    // entry points and device operations validate through validate_recipe_geometry below. Dense/joint
    // also mirror it in recipe_dense_q_tiles / recipe_dense_k_tiles (sdpa_recipe.cpp).
    if (q_tiles == 0 || k_tiles == 0 || d_tiles == 0) {
        return std::string("recipes require tile-aligned, nonzero Q/K chunks and head dims");
    }
    // Every recipe on every op: any tile-aligned Q chunk up to the recurrent-state arrays (32 tile rows), any
    // tile-aligned K chunk and head dim; L1 fit is recipe_l1_bytes.
    if (q_tiles > 32) {
        return fmt::format("Q chunk {} exceeds 1024 rows", q_tiles * kTile);
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

uint32_t recipe_mask_group_rows(const PrecisionPolicy& policy, [[maybe_unused]] uint32_t q_tiles) {
    return policy.fp32_destination ? 1 : 2;
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
    const uint64_t q_slot = uint64_t{q_tiles} * d_tiles * kBf16Tile;
    // The factories' L1 fallback drops the fused chunks' CBs, except for an odd STANDARD Q chunk (recipe_drop_fused).
    auto droppable_fused = [&](uint32_t rows, const RecipeBuild& build) -> uint64_t {
        return rows % 2 != 0 && policy.selection.recipe == Recipe::B ? 0 : build.fused_cb_bytes;
    };
    switch (op) {
        case RecipeOp::Dense:
        case RecipeOp::Joint: {
            const uint32_t rows =
                recipe_compute_q_tiles(policy, q_tiles, k_tiles, context.mask_page_bytes > 0, context.extra_bytes > 0);
            const auto build = recipe_build(policy, rows, k_tiles, d_tiles, context.vd_tiles);
            // attn_mask CB: two row groups when they fit, one otherwise (sdpa_recipe.cpp). The minimum layout
            // also drops the fused chunks' CBs (check_recipe_l1_fit).
            const uint64_t mask_group =
                uint64_t{recipe_mask_group_rows(policy, q_tiles)} * k_tiles * context.mask_page_bytes;
            return {
                build.cb_bytes + 2 * mask_group + context.extra_bytes,
                build.cb_bytes - droppable_fused(rows, build) + mask_group + context.extra_bytes};
        }
        case RecipeOp::Ring: {
            const auto build = recipe_build(policy, q_tiles, k_tiles, d_tiles);
            const uint64_t bytes = build.cb_bytes + kRingRecipeExtraBytes;
            // The factory's fallbacks: drop the fused chunks' CBs, then single-slot Q.
            return {bytes, bytes - droppable_fused(q_tiles, build) - q_slot};
        }
        case RecipeOp::ExpRing: {
            // The exp-ring factory keeps the recipe layout with a single Q slot, plus the 64 B live-length
            // mailbox of a device-tensor logical_n (counted unconditionally).
            const auto build = recipe_build(policy, q_tiles, k_tiles, d_tiles);
            const uint64_t bytes = build.cb_bytes - q_slot + 64;
            return {bytes, bytes - droppable_fused(q_tiles, build)};  // the factory drops the fused chunks' CBs to fit
        }
    }
    TT_THROW("Unknown SDPA recipe op");
}

bool recipe_program_fits(const RecipeBlockingProblem& p, uint32_t q_tiles, uint32_t k_tiles) {
    // A program's RISC images, runtime args, semaphores and CB configs share one 70656 B kernel config buffer per core.
    // Measured on one P150b (program bytes; BF16/BFP8 K/V; D64, D96, D128, D256; K128-K1024; Q128-Q544; with and
    // without a joint segment, attention sink, K tail, attn_mask, causal / sliding-window / paged key range), only
    // STANDARD's fused kernel comes near it. Each of these adds to its plain dense program (BFP8 Q256/K256: D64
    // 67936 B, D128 66960 B, D96 69744 B; BF16 K/V ~2.1 KB less): an odd Q chunk of 7+ tiles (a single-row last group)
    // 0.9-1.9 KB, a K tail 0.9 KB, a joint segment 2.2 KB (both 0.7 KB more with an odd Q chunk), an attention sink
    // 2.5 KB, a key range with a sink 2.7 KB. Excluded, as measured over or within 20 B of the limit (worst: BFP8 D96
    // Q224/K256 joint 73504 B; D64 72576 B; D128 Q224/K256 joint 71344 B, which op-chosen joint blocking hit):
    // - an odd Q chunk of 7+ tiles with a joint segment, a sink or a K tail, or with a head dim of odd tile count
    //   (one-tile PV subblocks: D96 BFP8 Q224 and Q288, K256 and K512, 69584-70640 B without features);
    // - packed K/V with a head dim of odd tile count and a joint segment, a sink or a K tail (D96 BFP8 Q256/K256:
    //   71840, 72272, 70656 B).
    // The tightest admitted programs: BFP8 D64 Q256/K256 with a sink 70368 B (70384 B with a key range), BF16 D96
    // Q256/K256 with a sink 70240 B, BFP8 D64 Q224/K256 dense 69728 B. FAST peaks at 68928 B (BFP8 Q224/K512 causal
    // with a sink and window), BALANCED and ACCURATE at 64.1 KB. Ring and exp ring programs are not covered.
    if ((p.op != RecipeOp::Dense && p.op != RecipeOp::Joint) || p.policy.selection.recipe != Recipe::B) {
        return true;
    }
    // An attn_mask or a one-tile QK subblock runs the unfused kernel, which pads an odd chunk and is far smaller.
    const bool attn_mask = p.mask_page_bytes > 0 && !p.key_range;
    if (attn_mask || recipe_subblock_width(k_tiles) < 2) {
        return true;
    }
    const bool k_tail = p.k_rows_unaligned || (p.k_rows + p.joint_k_rows) % (k_tiles * kTile) != 0;
    const bool feature = p.op == RecipeOp::Joint || p.attention_sink || k_tail;
    const bool narrow_pv = (p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles) % 2 != 0;
    if (q_tiles % 2 != 0 && q_tiles >= 7 && (feature || narrow_pv)) {
        return false;
    }
    return !(narrow_pv && feature && p.policy.selection.kv_storage != KVStorage::BF16);
}

namespace {
// Key-range work beyond the block roofline, fitted on one P150b (op time of 120 blockings, Q128-640 x K128-640:
// causal 10 heads D128, 16 heads D64, 32/8 heads D128; sliding window 1024 with 10 heads D128 and 16/8 heads D256;
// 8192 rows; STANDARD, ACCURATE, FAST BFP8; the pick is within 1.19x of the best measured blocking everywhere,
// within 1.11x but for one shape): each K chunk also costs kKeyRangeStream x c x k_tiles x d_tiles / 4 (streaming
// its K/V, which key-range cores read for themselves or pass along), each edge K chunk kKeyRangeEdge x c x q_tiles x
// k_tiles (the additive mask; STANDARD and FAST also lose the fused chunk), each Q chunk kKeyRangeJob x c (its first
// K chunk on the unfused path, Q load and normalization).
constexpr double kKeyRangeStream = 2.0;
constexpr double kKeyRangeEdge = 1.0;
constexpr double kKeyRangeJob = 200.0;

// Modeled work of the busiest core of a key-range call: each Q chunk costs its K chunks (mirroring
// RecipeKeyRange::chunks in dataflow/recipe_key_range.hpp: causal and sliding-window rows at q_offset + row), and all
// heads' Q chunks are dealt to `cores` in snake order by cost (run_recipe_segments: recipe_snake_count).
double key_range_makespan(
    const RecipeBlockingProblem& p, uint32_t q_tiles, uint32_t k_tiles, uint32_t cores, double block, double c) {
    const uint32_t q_chunk = q_tiles * kTile, k_chunk = k_tiles * kTile;
    const uint32_t k_chunks = div_up(p.k_rows, k_chunk);
    auto keys = [&](uint32_t q) {
        uint32_t lo = 0, hi = p.k_rows;
        if (p.causal) {
            hi = std::min(hi, q + 1);
        }
        if (p.sliding_window > 0) {
            if (p.causal) {
                lo = q + 1 >= p.sliding_window ? q + 1 - p.sliding_window : 0;
            } else {
                const uint32_t half = p.sliding_window / 2;
                lo = q >= half ? q - half : 0;
                hi = std::min(hi, q + half + 1);
            }
        }
        return std::pair{std::min(lo, hi), hi};
    };
    std::vector<double> jobs;
    for (uint32_t row0 = 0; row0 < p.q_rows; row0 += q_chunk) {
        const uint32_t row_end = std::min(row0 + q_chunk, p.q_rows);
        const auto [lo_first, hi_first] = keys(p.q_offset + row0);
        const auto [lo_last, hi_last] = keys(p.q_offset + row_end - 1);
        const uint32_t first = std::min(lo_first / k_chunk, k_chunks - 1);
        const uint32_t end = std::max(div_up(hi_last, k_chunk), first + 1);
        uint32_t full_begin = div_up(lo_last, k_chunk);
        uint32_t full_end = hi_first / k_chunk;
        if (full_end < full_begin || hi_first <= lo_last) {
            full_end = full_begin;
        }
        full_begin = std::clamp(full_begin, first, end);
        full_end = std::clamp(full_end, full_begin, end);
        const uint32_t edges = (end - first) - (full_end - full_begin);
        const double k_block = block + kKeyRangeStream * c * k_tiles * p.d_tiles / 4.0;
        jobs.push_back((end - first) * k_block + edges * kKeyRangeEdge * c * q_tiles * k_tiles + kKeyRangeJob * c);
    }
    // Every head has the same Q chunk costs: sorted entry s is job s / heads of the descending list.
    std::sort(jobs.begin(), jobs.end(), std::greater<>());
    const uint32_t heads = p.batch * p.q_heads;
    const uint64_t total = uint64_t{heads} * jobs.size();
    std::vector<double> load(cores, 0.0);
    for (uint64_t s = 0; s < total; ++s) {
        const uint64_t round = s / cores, position = s % cores;
        load[round % 2 == 0 ? position : cores - 1 - position] += jobs[s / heads];
    }
    return *std::max_element(load.begin(), load.end());
}
}  // namespace

std::vector<RecipeBlocking> recipe_blocking_candidates(const RecipeBlockingProblem& p) {
    std::vector<RecipeBlocking> candidates;
    const uint32_t batch_heads = p.batch * p.q_heads;
    if (batch_heads == 0 || p.q_rows == 0 || p.k_rows == 0 || p.grid.x == 0 || p.grid.y == 0) {
        return candidates;
    }
    // A chunk longer than the sequence only adds padding; the default sizes stay in range.
    const bool dense = p.op == RecipeOp::Dense || p.op == RecipeOp::Joint;
    const uint32_t q_cap = std::min(kRecipeSearchMaxQTiles, std::max(div_up(p.q_rows + p.joint_q_rows, kTile), 10u));
    // Padded K blocks are costed, so short K picks short chunks. Fused FAST amortizes its per-chunk
    // work (saturation check, fold, PV pieces) over longer K chunks: up to K1024 when L1 allows.
    const uint32_t k_cap = p.policy.selection.recipe == Recipe::E ? 2 * kRecipeSearchMaxKTiles : kRecipeSearchMaxKTiles;
    const uint32_t q_floor = std::min(kRecipeSearchMinQTiles, div_up(p.q_rows + p.joint_q_rows, kTile));
    // A sliding window bounds each Q chunk's keys, so K chunks down to 128 rows waste fewer masked keys (fitted with
    // the key-range terms, kKeyRangeEdge).
    const uint32_t k_min = p.key_range && p.sliding_window > 0 ? kRecipeSearchMinKTiles / 2 : kRecipeSearchMinKTiles;
    const uint32_t k_floor = std::min(k_min, div_up(p.k_rows + p.joint_k_rows, kTile));
    const auto q_range = tile_range(p.fixed_q_tiles, q_floor, q_cap);
    const auto k_range = tile_range(p.fixed_k_tiles, k_floor, k_cap);
    for (auto qi = q_range.rbegin(); qi != q_range.rend(); ++qi) {  // ascending Q
        const uint32_t qt = *qi;
        bool any_fit = false;
        for (auto ki = k_range.rbegin(); ki != k_range.rend(); ++ki) {  // ascending K
            const uint32_t kt = *ki;
            if (!recipe_geometry_supported(p.op, p.policy, qt, kt, p.d_tiles) ||
                (dense && !recipe_program_fits(p, qt, kt))) {
                continue;
            }
            // Ring / exp ring round STANDARD's odd Q chunk up to the next even one (sdpa.cpp); the even
            // candidate is costed on its own.
            if (!dense && qt % 2 != 0 && recipe_compute_q_tiles(p.policy, qt, kt) != qt) {
                continue;
            }
            // Dense/joint L1 grows with Q and K: stop at the first K that does not fit.
            if (dense &&
                recipe_l1_bytes(
                    p.op,
                    p.policy,
                    qt,
                    kt,
                    p.d_tiles,
                    {.mask_page_bytes = p.mask_page_bytes, .extra_bytes = p.extra_l1_bytes, .vd_tiles = p.vd_tiles})
                        .minimum > p.l1_bytes) {
                break;
            }
            any_fit = true;
            const uint32_t q_chunk = qt * kTile;
            const uint32_t k_chunk = kt * kTile;
            const RecipeBuild build = recipe_build(p.policy, qt, kt, p.d_tiles, dense ? p.vd_tiles : 0);
            const double block = block_cost(p.policy, qt, kt, p.d_tiles, build, dense, p.key_range);
            // K tiles of a Q chunk's first K/V block: the chunk, or the whole (primary) sequence when shorter.
            const uint32_t first_k_tiles = std::min(kt, std::max(1u, div_up(p.k_rows, kTile)));
            // `blocks`: the busiest core's (Q chunk, K chunk) blocks (jobs x K chunks unless a key range says less).
            // `blocks`: the busiest core's (Q chunk, K chunk) blocks (jobs x K chunks unless a key range says less);
            // `extra`: key-range work outside the blocks.
            auto admit = [&](uint32_t jobs,
                             uint32_t k_blocks,
                             CoreCoord grid,
                             const RecipeL1Context& context,
                             std::optional<double> blocks = std::nullopt,
                             double extra = 0.0) {
                const auto l1 = recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, context);
                if (l1.minimum > p.l1_bytes) {
                    return;
                }
                double cost = blocks.value_or(static_cast<double>(jobs) * k_blocks) * block + extra +
                              block_overhead(p.policy, qt, first_k_tiles, p.d_tiles, jobs);
                if (l1.preferred > p.l1_bytes) {
                    cost *= kFallbackPenalty;
                }
                candidates.push_back({q_chunk, k_chunk, grid, cost, jobs, l1});
            };
            switch (p.op) {
                case RecipeOp::Dense:
                case RecipeOp::Joint: {
                    // run_recipe_segments: one KV-forwarding chain per batch/head, jobs split evenly; with more
                    // batch/heads than cores, every head's Q chunks split evenly over the grid.
                    const uint32_t cores = p.grid.x * p.grid.y;
                    const uint32_t jobs_per_head = div_up(p.q_rows + p.joint_q_rows, q_chunk);
                    const uint32_t chain =
                        std::min({jobs_per_head, cores / batch_heads, p.max_cores_per_head_batch});
                    if (p.key_range) {
                        // run_recipe_segments: all heads' Q chunks dealt over the grid in snake order by cost.
                        const uint32_t total_jobs = batch_heads * jobs_per_head;
                        const uint32_t used = std::min(cores, total_jobs);
                        const uint32_t k_blocks = div_up(p.k_rows, k_chunk);
                        const uint32_t jobs = div_up(total_jobs, used);
                        const double c = block_cost_model(p.policy).c;
                        // Windowed segments (unknown on the host) cost every K chunk.
                        const double work =
                            p.causal || p.sliding_window > 0
                                ? key_range_makespan(p, qt, kt, used, block, c)
                                : jobs * (k_blocks * (block + kKeyRangeStream * c * kt * p.d_tiles / 4.0) +
                                          kKeyRangeJob * c);
                        admit(
                            jobs,
                            k_blocks,
                            p.grid,
                            RecipeL1Context{
                                .mask_page_bytes = p.mask_page_bytes,
                                .extra_bytes = p.extra_l1_bytes,
                                .vd_tiles = p.vd_tiles},
                            0.0,
                            work);
                        break;
                    }
                    if (chain == 0 && p.max_cores_per_head_batch == 0) {
                        break;
                    }
                    const uint32_t k_blocks = div_up(p.k_rows + p.joint_k_rows, k_chunk);
                    const double reread =
                        chain == 0 ? kKvReread * static_cast<double>(batch_heads) * jobs_per_head * k_blocks / cores *
                                         block_cost_model(p.policy).bw * kt * p.d_tiles / 4.0
                                   : 0.0;
                    admit(
                        chain == 0 ? div_up(batch_heads * jobs_per_head, cores) : div_up(jobs_per_head, chain),
                        k_blocks,
                        p.grid,
                        RecipeL1Context{
                            .mask_page_bytes = p.mask_page_bytes,
                            .extra_bytes = p.extra_l1_bytes,
                            .vd_tiles = p.vd_tiles},
                        std::nullopt,
                        reread);
                    break;
                }
                case RecipeOp::Ring: {
                    // ring_joint_sdpa_program_factory.cpp: all Q chunks of all heads split evenly over
                    // the worker grid; every ring iteration streams the local KV shard, the joint KV once.
                    const uint32_t cores = p.grid.x * p.grid.y;
                    const uint32_t q_chunks = div_up(p.q_rows, q_chunk) + div_up(p.joint_q_rows, q_chunk);
                    const uint32_t jobs = div_up(batch_heads * q_chunks, cores);
                    const uint32_t k_blocks = p.ring_size * div_up(p.k_rows, k_chunk) + div_up(p.joint_k_rows, k_chunk);
                    admit(jobs, k_blocks, p.grid, RecipeL1Context{});
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
                        const CoreCoord grid(p.exp_mux_on_bottom_row ? cols : cols + 1, p.grid.y);
                        admit(passes, k_blocks, grid, RecipeL1Context{});
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
    const Tensor* attn_mask,
    uint64_t reserved_l1_bytes,
    const RecipeKeyRange* key_range,
    const RecipeDenseOptions* options,
    bool chunks_are_hints) {
    if ((!recipe_blocking_requested(program_config) && !chunks_are_hints) || q.storage_type() != StorageType::DEVICE) {
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
    // A paged cache's K length is its blocks per sequence x block size.
    problem.k_rows = key_range ? div_up(recipe_k_rows(k, *key_range), kTile) * kTile : k.padded_shape()[2];
    problem.joint_q_rows = joint_q ? joint_q->padded_shape()[2] : 0;
    problem.joint_k_rows = joint_k ? joint_k->padded_shape()[2] : 0;
    if (options && options->head_dim_v) {
        problem.vd_tiles = div_up(options->head_dim_v, kTile);
    }
    problem.attention_sink = options && options->attention_sink.has_value();
    problem.k_rows_unaligned = (key_range ? recipe_k_rows(k, *key_range) : k.logical_shape()[2]) % kTile != 0 ||
                               (joint_k && joint_k->logical_shape()[2] % kTile != 0);
    const uint64_t free_l1 = free_l1_below_live_buffers(*device);
    problem.l1_bytes = free_l1 > reserved_l1_bytes ? free_l1 - reserved_l1_bytes : 0;
    problem.mask_page_bytes = attn_mask ? attn_mask->buffer()->page_size() : 0;
    if (key_range && key_range->active()) {
        problem.mask_page_bytes = 576;  // BFP4 mask tiles
        problem.extra_l1_bytes = recipe_key_range_extra_bytes(*key_range);
        problem.key_range = true;
        problem.causal = key_range->causal;
        problem.sliding_window = key_range->sliding_window;
        problem.q_offset = key_range->q_offset;
        if (key_range->q_offset_tensor) {
            // The start is read on device (trace-safe chunked prefill), so one blocking serves every start: cost the
            // latest one the K/V length allows, the most expensive (tt_transformers Q2048 at 6144 of 8192 keys:
            // Q256/K512 3.97 ms, the choice for a start of 0 5.22 ms).
            problem.q_offset = problem.k_rows > problem.q_rows ? problem.k_rows - problem.q_rows : 0;
        }
    }
    if (options) {
        problem.extra_l1_bytes += recipe_dense_options_extra_bytes(*options);
    }
    if (key_range && key_range->q_slab_rows) {
        // Ring-distributed SDPA: two Q slabs, each of whole Q chunks.
        problem.q_rows = 2 * key_range->q_slab_rows;
        std::optional<RecipeBlocking> choice;
        if (!invalid_fixed(config)) {
            for (const auto& candidate : recipe_blocking_candidates(problem)) {
                if (key_range->q_slab_rows % candidate.q_chunk_size == 0) {
                    choice = candidate;
                    break;
                }
            }
        }
        return apply_choice(config, choice, problem, "ring-distributed");
    }
    const char* op_name = joint_q ? "joint" : "dense";
    if (chunks_are_hints) {
        // Routed dense and joint calls choose their blocking on the whole compute grid: the caller's chunks and grid
        // were tuned for the legacy kernels, and the chooser beats them (tt_transformers causal S8192 on its 8x8 grid:
        // 7.77 vs 14.49 ms, Qwen3-VL vision 2.03 vs 3.50; Flux-style joint 1.52 vs 1.58, qwen_image joint equal).
        config.compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        problem.grid = config.compute_with_storage_grid_size;
        problem.fixed_q_tiles = 0;
        problem.fixed_k_tiles = 0;
        config.q_chunk_size = 0;
        config.k_chunk_size = 0;
        return apply_choice(config, choose_recipe_blocking(problem), problem, op_name);
    }
    const auto choice = invalid_fixed(config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(config, choice, problem, op_name);
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
