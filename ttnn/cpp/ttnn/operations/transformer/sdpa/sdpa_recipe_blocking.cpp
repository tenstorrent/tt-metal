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
constexpr uint32_t kFrozenQTiles = 8;
constexpr uint32_t kFrozenKTiles = 16;
// Prefer the frozen, bit-for-bit qualified Q256/K512 geometry when its modeled cost is within
// this fraction of the cheapest candidate: the cost model is not that precise.
constexpr double kFrozenPreference = 0.02;
// Ring layouts whose preferred (double-buffered Q) layout does not fit lose Q prefetch.
constexpr double kFallbackPenalty = 1.03;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

bool is_paired(const PrecisionPolicy& policy) { return policy.recurrent_state == RecurrentState::CompensatedBF16; }

// Per (Q chunk, K chunk) block cost model, a roofline in units of one D128 QK+PV tile product:
//   compute   = c * (q_tiles * k_tiles * d_tiles / 4 + ck * q_tiles * d_tiles / 4)
//   bandwidth = bw * k_tiles * d_tiles / 4
//   cost      = max(compute, bandwidth)
// `ck` is the per-row, per-K-chunk softmax state work (so short K chunks cost more per key) and
// `bw` the Q tile count below which streaming the K/V chunk into the core dominates (so short Q
// chunks cost more per row). Per variant, fitted to the matched-chunk trace timings in
// docs/sdpa_precision_qualification.md (single P150b, 10 heads, 8192 x 8192, D128, full grid):
// `c` from Q256/K512, `ck` from Q256/K256 (it also predicts Q320/K256 within 2%), `bw` from
// Q128/K512.
struct BlockCostModel {
    double c;
    double ck;
    double bw;
};

BlockCostModel block_cost_model(const PrecisionPolicy& policy) {
    switch (policy.selection.recipe) {
        case Recipe::A: return {1.000, 1.49, 5.99};
        case Recipe::B: return {1.216, 5.28, 7.35};
        case Recipe::C: return {1.878, 3.58, 9.46};
        case Recipe::D: return {2.436, 3.14, 11.75};
        case Recipe::E:
            switch (policy.selection.kv_storage) {
                case KVStorage::BF16: return {1.040, 7.19, 7.75};
                case KVStorage::BFP8: return {1.036, 7.02, 6.03};
                case KVStorage::BFP4: return {1.028, 7.96, 3.50};
            }
    }
    TT_THROW("Unknown SDPA precision recipe");
}

// Build-flag facts of a geometry, read from the recipe program the factory would build.
struct RecipeBuild {
    uint64_t cb_bytes = 0;
    bool size_optimized = false;  // SDPA_RECIPE_SIZE_OPTIMIZED: pack thread built at -Os
    uint32_t qk_width = 4;        // SDPA_RECIPE_QK_W
    uint32_t pv_width = 4;        // SDPA_RECIPE_PV_W
};

// Compute slowdowns not in the fitted model: a size-optimized pack thread (odd or new paired
// geometries) and narrower matmul subblocks.
constexpr double kSizeOptimizedPenalty = 1.08;
double subblock_penalty(uint32_t width) { return width >= 4 ? 1.0 : width == 2 ? 1.10 : 1.30; }

double block_cost(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles, const RecipeBuild& build) {
    const auto m = block_cost_model(policy);
    const double d = d_tiles / 4.0;
    double compute = m.c * (static_cast<double>(q_tiles) * k_tiles * d + m.ck * q_tiles * d);
    compute *= std::max(subblock_penalty(build.qk_width), subblock_penalty(build.pv_width));
    if (build.size_optimized) {
        compute *= kSizeOptimizedPenalty;
    }
    return std::max(compute, m.bw * k_tiles * d);
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
    for (const auto& [name, value] : program.kernels.front().defines) {
        if (name == "SDPA_RECIPE_SIZE_OPTIMIZED") {
            build.size_optimized = true;
        } else if (name == "SDPA_RECIPE_QK_W") {
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

uint64_t recipe_cb_bytes(const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
    return recipe_build(policy, q_tiles, k_tiles, d_tiles).cb_bytes;
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
// signal page and derived-geometry mailbox. Mirrors that factory until FAST moves to the recipe
// ring factory (docs/sdpa_recipe_consolidation.md task 4).
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
    (void)policy;
    // Mirrors recipe_dense_q_tiles / recipe_dense_k_tiles (dense, joint) and recipe_q_tiles /
    // recipe_k_tiles plus the ring / exp-ring device operation validation (sdpa_recipe.cpp,
    // sdpa.cpp, *_device_operation.cpp). Keep those checks and this function in lockstep; the
    // chooser only consults this one.
    if (q_tiles == 0 || k_tiles == 0 || d_tiles == 0) {
        return std::string("recipes require tile-aligned, nonzero Q/K chunks and head dims");
    }
    if (op == RecipeOp::Dense || op == RecipeOp::Joint) {
        if (q_tiles > 32) {
            return fmt::format("Q chunk {} exceeds 1024 rows", q_tiles * kTile);
        }
        return std::nullopt;
    }
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
            const uint64_t bytes = recipe_cb_bytes(policy, q_tiles, k_tiles, d_tiles);
            return {bytes, bytes};
        }
        case RecipeOp::Ring: {
            if (fast) {
                const uint64_t bytes =
                    legacy_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.q_blocks_per_worker > 1 ? 2 : 1);
                return {bytes, bytes};
            }
            const uint64_t bytes = recipe_cb_bytes(policy, q_tiles, k_tiles, d_tiles) + kRingRecipeExtraBytes;
            return {bytes, bytes - q_slot};  // the factory's single-slot Q fallback
        }
        case RecipeOp::ExpRing: {
            if (fast) {
                return {
                    legacy_exp_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.passes, true),
                    legacy_exp_ring_fast_bytes(q_tiles, k_tiles, d_tiles, context.passes, false)};
            }
            // The exp-ring factory keeps the recipe layout with a single Q slot.
            const uint64_t bytes = recipe_cb_bytes(policy, q_tiles, k_tiles, d_tiles) - q_slot;
            return {bytes, bytes};
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
    // A chunk longer than the sequence only adds padding; the qualified sizes stay in range.
    const bool dense = p.op == RecipeOp::Dense || p.op == RecipeOp::Joint;
    const uint32_t q_cap = std::min(kRecipeSearchMaxQTiles, std::max(div_up(p.q_rows + p.joint_q_rows, kTile), 10u));
    const uint32_t k_cap = std::min(kRecipeSearchMaxKTiles, std::max(div_up(p.k_rows + p.joint_k_rows, kTile), 16u));
    const auto q_range = tile_range(p.fixed_q_tiles, kRecipeSearchMinQTiles, q_cap);
    const auto k_range = tile_range(p.fixed_k_tiles, 1, k_cap);
    for (auto qi = q_range.rbegin(); qi != q_range.rend(); ++qi) {  // ascending Q
        const uint32_t qt = *qi;
        bool any_fit = false;
        for (auto ki = k_range.rbegin(); ki != k_range.rend(); ++ki) {  // ascending K
            const uint32_t kt = *ki;
            if (!recipe_geometry_supported(p.op, p.policy, qt, kt, p.d_tiles)) {
                continue;
            }
            // Dense/joint L1 grows with Q and K: stop at the first K that does not fit.
            if (dense && recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles).minimum > p.l1_bytes) {
                break;
            }
            any_fit = true;
            const uint32_t q_chunk = qt * kTile;
            const uint32_t k_chunk = kt * kTile;
            // FAST ring / exp ring keep their legacy compute; everything else builds the recipe program.
            const bool recipe_compute = dense || p.policy.selection.recipe != Recipe::A;
            const double block = block_cost(
                p.policy, qt, kt, p.d_tiles, recipe_compute ? recipe_build(p.policy, qt, kt, p.d_tiles) : RecipeBuild{});
            auto admit = [&](uint32_t jobs, uint32_t k_blocks, CoreCoord grid, const RecipeL1Context& context) {
                const auto l1 = recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, context);
                if (l1.minimum > p.l1_bytes) {
                    return;
                }
                double cost = static_cast<double>(jobs) * k_blocks * block;
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
                        RecipeL1Context{});
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
                    // An explicit Q chunk keeps the caller's grid; otherwise the op may narrow it.
                    const uint32_t min_cols = p.fixed_q_tiles != 0 ? max_cols : 2;
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
    // Cheapest first; ties prefer larger K, then larger Q, then a wider grid.
    std::stable_sort(candidates.begin(), candidates.end(), [](const RecipeBlocking& a, const RecipeBlocking& b) {
        return std::make_tuple(a.cost, -static_cast<int64_t>(a.k_chunk_size), -static_cast<int64_t>(a.q_chunk_size),
                               -static_cast<int64_t>(a.grid.x)) <
               std::make_tuple(b.cost, -static_cast<int64_t>(b.k_chunk_size), -static_cast<int64_t>(b.q_chunk_size),
                               -static_cast<int64_t>(b.grid.x));
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
            if (candidate.cost > choice->cost * (1.0 + kFrozenPreference)) {
                break;
            }
            if (candidate.q_chunk_size == kFrozenQTiles * kTile && candidate.k_chunk_size == kFrozenKTiles * kTile) {
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
    TT_FATAL(
        !program_config || (program_config->q_chunk_size != 0 && program_config->k_chunk_size != 0),
        "SDPA q_chunk_size/k_chunk_size of 0 (op-selected blocking) requires an explicit precision recipe");
}

namespace {

uint64_t unreserved_l1(tt::tt_metal::distributed::MeshDevice& device) {
    return device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

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
        // Nothing fits: fall back to the frozen geometry so the op's validation explains why.
        config.q_chunk_size = auto_q ? kFrozenQTiles * kTile : config.q_chunk_size;
        config.k_chunk_size = auto_k ? kFrozenKTiles * kTile : config.k_chunk_size;
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

std::optional<SDPAProgramConfig> resolve_dense_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const Tensor* joint_q,
    const Tensor* joint_k,
    const std::optional<SDPAProgramConfig>& program_config) {
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
    problem.l1_bytes = unreserved_l1(*device);
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
    problem.l1_bytes = unreserved_l1(*q.device());
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
    // The exp-ring factory budgets CBs below the lowest live L1 buffer (global semaphores and
    // persistent buffers occupy the top of L1 in a pipeline).
    auto* device = q.device();
    const auto lowest = device->lowest_occupied_compute_l1_address();
    const uint64_t top = lowest.has_value() ? static_cast<uint64_t>(*lowest) : device->l1_size_per_core();
    problem.l1_bytes = top - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const auto choice = invalid_fixed(program_config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(program_config, choice, problem, "exp ring");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
