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

#include "sdpa_recipe_program.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
namespace {

constexpr uint32_t kTile = 32;
constexpr uint32_t kBf16Tile = 2048;
constexpr uint32_t kFp32Tile = 4096;
constexpr uint32_t kDefaultQTiles = 8;
constexpr uint32_t kDefaultKTiles = 16;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------------------------------------------
// Cost model. Every cost is in Tensix cycles of the busiest core: documented hardware rates times the work the recipe
// kernels do per (Q chunk, K chunk) block. The few quantities not documented in-repo are named assumptions below, each
// with the measurement that checks it; one of them, the K/V chain's stream rate, is read off SDPA op times.
//
//   block    = max(math engine, pack thread, unpacker) + the row groups' serial pack -> unpack round trips
//   Q chunk  = its blocks + normalization (+ the fused recipes' first K chunk on the reduce path)
//              against its K/V (and Q, output, mask) traffic: DRAM shared by every reading core, one reader's
//              latency-bound batches, and the NoC (two K/V slots overlap a transfer with compute, one serializes it)
//   core     = DRAM fill (first Q chunk and K/V block) + its Q chunks + DRAM drain (last output chunk)
//
// It reproduces the measured single-core throughput of every recipe at Q256/K512 within 8%, the full-grid op times
// of 22 P150b blockings (causal, sliding window, masked, noncausal) within 25% and of 20 b04u08 blockings within 20%,
// ranking every measured pair of blockings of one shape the same way as the measurement but one (Wan 720p FAST
// Q288/K512 vs Q256/K512, 6% apart).
// ---------------------------------------------------------------------------------------------------------------

// Tensix quantities, the same on Wormhole and Blackhole (tech_reports/matrix_engine/matrix_engine.md).
// The matrix engine computes 8x16 x 16x16 per cycle: a 32x32x32 tile product takes 16 cycles per fidelity phase
// (LoFi 1, HiFi2 2, HiFi3 3, HiFi4 4; tech_reports/GEMM_FLOPS/GEMM_FLOPS.md "Cycle_per_tile").
constexpr double kFpuCyclesPerPhase = 16.0;
// 16x16 reduce per cycle (4 cycles per tile) and 8x16 eltwise per cycle (8 cycles per tile).
constexpr double kFpuReduceCycles = 4.0;
constexpr double kFpuEltwiseCycles = 8.0;
// The LLK SFPU loop: 8 iterations of 32 datums per face, 4 faces per tile; one SFPU instruction per iteration-cycle.
constexpr double kSfpuIterationsPerTile = 32.0;
// Assumption (not documented in-repo): the packer writes one 64 B word per cycle (a BF16 tile in 32 cycles). It
// matches the LOW_PRECISION study's measured ~140-cycle pack thread per fused score tile (a 64-cycle exp + two packs).
constexpr double kPackBytesPerCycle = 64.0;
// Assumption: an unpacker fills a BF16 source tile per LoFi tile product (16 cycles; the tt-llk perf_matmul test
// sustains ~17 cycles per tile at LoFi with one operand reused).
constexpr double kUnpackBytesPerCycle = 128.0;
// Assumption: a matmul output subblock costs one DEST section handoff and MOP re-arm (the r6 STANDARD measurements,
// 1.10x at QK subblock width 2 and 1.30x at width 1, are one ~64-cycle handoff per subblock).
constexpr double kDestHandoffCycles = 64.0;
// FAST re-programs the packer per QK subblock (format, row width, ReLU, L1 accumulate; STANDARD keeps that setup across
// subblocks, recipe_fused_chunk.inc): one packer drain (a BF16 tile, kBf16Tile / kPackBytesPerCycle) each.
constexpr double kFastPackSetups = 4.0;
// Assumption: one pack -> unpack round trip (packer drain to L1, semaphore, unpacker wait). A row group's per-chunk
// state steps (saturation check, sum fold, max, rescale) each wait for one; the pipeline cannot overlap them.
// Consistent with Wormhole STANDARD on one core (Sq1024 Sk4096 D128): Q128/K256 0.865 vs Q128/K512 1.029 TF/s.
constexpr double kSyncCycles = 256.0;
// Candidates whose costs are within this fraction of the cheapest are ties: the model ranks two blockings of one shape
// only to about 10% (b04u08 pairs: Wan 720p FAST Q288/K512 vs Q256/K512 predicted 0.96, measured 1.06; bge_m3
// ACCURATE Q512/K256 vs Q256/K512 1.13 vs 1.19; 4096-row STANDARD Q160/K512 vs Q192/K512 1.36 vs 1.30). Half of that
// keeps the measured wins (20-28% on causal GQA shapes) while changing the fewest other picks.
constexpr double kTieTolerance = 0.05;

// Architecture quantities.
struct ArchQuantities {
    double dram_bytes_per_cycle;  // achievable interleaved DRAM read bandwidth over the Tensix clock
    double noc_bytes_per_cycle;   // NoC word (flit) per cycle into one core
    // Measured assumption (not derivable from the quantities here): the K/V bytes per cycle a forwarding chain of two
    // or more cores sustains (receive, forward, then publish each chunk; reader_recipe.cpp). Blackhole b04u08, 10
    // heads noncausal: FAST BF16 Q128/K896 2.82 ms and Q256/K512 1.39 ms, STANDARD 4096 rows Q160/K512 0.73 ms and
    // Q192/K512 0.56 ms all stream at ~7 B/cycle; Wormhole is scaled by its NoC word (unmeasured).
    double chain_stream_bytes_per_cycle;
    // noc_latencies.yaml (tt_metal/impl/experimental/noc_estimator, measured): one core reading n interleaved DRAM
    // pages of 512, 1024, 2048 and 4096 B and then waiting on a barrier (ONE_FROM_ALL, DRAM_INTERLEAVED, NoC 0),
    // for n = 1, 4 and 16, in cycles.
    std::array<std::array<double, 4>, 3> read_latency;
};
// Interleaved DRAM reads reach ~190 of 288 GB/s (tech_reports/LLMs/llms.md, "DRAM Interleaved").
constexpr double kInterleavedDramFraction = 190.0 / 288.0;
// Blackhole: 1.35 GHz (GEMM_FLOPS.md); 8 GDDR6 channels, ~512 GB/s (tech_reports/CCLs/CCL_bandwidth.md); 64 B NoC
// words (noc_parameters.h NOC_WORD_BYTES; tech_reports/PCIe_bandwidth/PCIe_bandwidth.md flit size).
constexpr ArchQuantities kBlackhole{
    .dram_bytes_per_cycle = 512e9 * kInterleavedDramFraction / 1.35e9,
    .noc_bytes_per_cycle = 64.0,
    .chain_stream_bytes_per_cycle = 7.0,
    .read_latency = {{{496, 512, 537, 576}, {589, 742, 629, 709}, {915, 921, 1042, 1521}}}};
// Wormhole: 1 GHz (GEMM_FLOPS.md); 12 GDDR6 channels, 288 GB/s (tech_reports/FlashAttention/FlashAttention.md);
// 32 B NoC words.
constexpr ArchQuantities kWormhole{
    .dram_bytes_per_cycle = 288e9 * kInterleavedDramFraction / 1.0e9,
    .noc_bytes_per_cycle = 32.0,
    .chain_stream_bytes_per_cycle = 3.5,
    .read_latency = {{{405, 429, 470, 557}, {507, 532, 661, 924}, {879, 927, 1457, 2511}}}};

const ArchQuantities& arch_quantities(const RecipeBlockingProblem& p) { return p.wormhole ? kWormhole : kBlackhole; }

double interpolate(const std::array<double, 4>& xs, const std::array<double, 4>& ys, double x) {
    if (x <= xs[0]) {
        return ys[0];
    }
    for (size_t i = 1; i < xs.size(); ++i) {
        if (x <= xs[i]) {
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
        }
    }
    return ys.back() * x / xs.back();
}

// Cycles for one core to read `pages` DRAM pages of `page_bytes`, `barrier` pages per read barrier (the reader's
// interval; 1 to 16).
double read_cycles(const ArchQuantities& arch, double pages, double page_bytes, uint32_t barrier) {
    constexpr std::array<double, 4> kPageBytes{512, 1024, 2048, 4096};
    std::array<double, 4> at{};
    for (size_t n = 0; n < 3; ++n) {
        at[n] = interpolate(kPageBytes, arch.read_latency[n], page_bytes);
    }
    const double batch = interpolate({1, 4, 16, 64}, {at[0], at[1], at[2], 4 * at[2]}, barrier);
    return pages / barrier * batch;
}

// reader_recipe.cpp: the chain head (and every non-key-range reader) waits on a barrier every 16 K/V tiles.
constexpr uint32_t kChainReadBarrier = 16;

uint32_t kv_tile_bytes(KVStorage storage) {
    switch (storage) {
        case KVStorage::BF16: return 2048;
        case KVStorage::BFP8: return 1088;
        case KVStorage::BFP4: return 576;
    }
    TT_THROW("Unknown SDPA K/V storage");
}

// Fidelity phases of the QK, PV and (FP32 recipes) denominator matmuls, and SFPU instructions per 32-datum row of the
// score exp (recipe_sfpu.hpp: the fast exp's two; BALANCED's grid two plus its quadratic refinement's four, ACCURATE's
// cubic five).
struct RecipeWork {
    double qk_phases;
    double pv_phases;
    double denominator_phases;
    double exp_instructions;
};

RecipeWork recipe_work(const PrecisionPolicy& policy) {
    switch (policy.selection.recipe) {
        case Recipe::B: return {2, 2, 0, 2};  // HiFi2 / HiFi2
        case Recipe::C: return {3, 2, 1, 6};  // HiFi3 QK (kBalancedQkFidelity), HiFi2 PV, LoFi denominator
        case Recipe::D: return {4, 4, 2, 7};  // HiFi4 / HiFi4, HiFi2 denominator
        case Recipe::E: return {1, 1, 0, 2};  // LoFi / LoFi
    }
    TT_THROW("Unknown SDPA precision recipe");
}

// Q tile rows per pipelined row group (recipe_fused_chunk.inc): dense Blackhole STANDARD runs Q chunks up to six tiles
// in one-row groups, the BF16 recipes pair rows otherwise; the FP32 recipes work row by row.
uint32_t group_rows(const RecipeBlockingProblem& p, uint32_t rows) {
    if (p.policy.fp32_destination) {
        return 1;
    }
    const bool dense = p.op == RecipeOp::Dense || p.op == RecipeOp::Joint;
    return p.policy.selection.recipe == Recipe::B && dense && rows <= 6 && !p.wormhole ? 1 : 2;
}

// Serial pack -> unpack round trips per row group and K chunk.
double syncs_per_group(const PrecisionPolicy& policy, bool fused, uint32_t vd) {
    if (fused) {
        // The saturation check (STANDARD checks two groups per unit, FAST every group: issue and verdict), the sum
        // fold and the QK / PV switch.
        return policy.selection.recipe == Recipe::B ? 3 : 4;
    }
    if (!policy.fp32_destination) {
        return 5;  // reduce path: row max, S - m and exp, row sums, correction, O and l rescale
    }
    // Row max, S - m, exp, denominator, correction, l rescale, then O rescaled two head-dim tiles per DEST pass.
    return 6 + div_up(vd, 2);
}

// Build-flag facts of a geometry, read from the recipe program the factory would build.
struct RecipeBuild {
    uint64_t cb_bytes = 0;
    uint64_t fused_cb_bytes = 0;  // CBs 29-31 of fused chunks, which a factory drops when they do not fit
    uint64_t kv_cb_bytes = 0;     // the K and V CBs (1, 2), which the ring factories resize
    uint32_t qk_width = 4;        // recipe_qk_w
    uint32_t pv_width = 4;        // recipe_pv_w
    bool fused = false;           // recipe_fused: fused STANDARD / FAST chunks
};

// The recipe's own circular buffers and build flags (recipe_compute_program), which dense, joint
// and the B-E ring and exp-ring program factories adopt as their compute layout.
RecipeBuild recipe_build(
    const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    uint32_t vd_tiles = 0,
    std::optional<KVStorage> v_storage = std::nullopt) {
    static std::mutex mutex;
    static std::map<std::tuple<uint8_t, uint8_t, uint8_t, uint32_t, uint32_t, uint32_t, uint32_t>, RecipeBuild> memo;
    const auto v_kind = v_storage.value_or(policy.selection.kv_storage);
    const auto key = std::make_tuple(
        static_cast<uint8_t>(policy.selection.recipe),
        static_cast<uint8_t>(policy.selection.kv_storage),
        static_cast<uint8_t>(v_kind),
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
    const auto program =
        recipe_compute_program(policy, core, 1, q_tiles, k_tiles, d_tiles, std::nullopt, vd_tiles, v_kind);
    RecipeBuild build{.cb_bytes = recipe_cb_bytes(program)};
    for (const auto& cb : program.cbs) {
        build.fused_cb_bytes += recipe_fused_cb(cb) ? cb.total_size : 0;
        const uint8_t index = cb.format_descriptors.front().buffer_index;
        build.kv_cb_bytes += index == 1 || index == 2 ? cb.total_size : 0;
    }
    const auto& args = program.kernels.front().named_compile_time_args;
    build.qk_width = named_arg(args, "recipe_qk_w");
    build.pv_width = named_arg(args, "recipe_pv_w");
    build.fused = named_arg(args, "recipe_fused") != 0;
    std::lock_guard lock(mutex);
    if (memo.size() > 65536) {
        memo.clear();
    }
    memo.emplace(key, build);
    return build;
}

// K/V tiles and bytes of one K chunk of `k_tiles`.
struct KvChunk {
    double tiles;
    double bytes;
};

KvChunk kv_chunk(const RecipeBlockingProblem& p, uint32_t k_tiles, bool dense) {
    const uint32_t vd = dense && p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles;
    const uint32_t kb = kv_tile_bytes(p.policy.selection.kv_storage);
    const KVStorage v_storage =
        dense ? p.v_storage.value_or(p.policy.selection.kv_storage) : p.policy.selection.kv_storage;
    const uint32_t vb = kv_tile_bytes(v_storage);
    return {
        static_cast<double>(k_tiles) * (p.d_tiles + vd),
        static_cast<double>(k_tiles) * (static_cast<double>(p.d_tiles) * kb + static_cast<double>(vd) * vb)};
}

// Cycles of one (Q chunk, K chunk) block's compute on one core: `rows` Q tile rows (recipe_compute_q_tiles) by
// `k_tiles` keys; `fused`: the fused STANDARD / FAST chunk, else the reduce path; `masked`: an additive mask
// (attn_mask, or a key range's edge chunk).
double block_compute(
    const RecipeBlockingProblem& p,
    uint32_t rows,
    uint32_t k_tiles,
    const RecipeBuild& build,
    bool dense,
    bool fused,
    bool masked) {
    const auto& policy = p.policy;
    const auto work = recipe_work(policy);
    const bool fp32 = policy.fp32_destination;
    const uint32_t vd = dense && p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles;
    const double d = p.d_tiles, v = vd, r = rows, k = k_tiles;
    const double kb = kv_tile_bytes(policy.selection.kv_storage);
    const double vb =
        kv_tile_bytes(dense ? p.v_storage.value_or(policy.selection.kv_storage) : policy.selection.kv_storage);
    const double state = fp32 ? kFp32Tile : kBf16Tile;  // score and state tile bytes
    const double scores = r * k;
    // Math engine: the matmuls; the fused chunk's one-phase m_ref inner step, or the reduce path's row max and S - m;
    // the FP32 recipes' denominator matmul and O = O * c + PV; the additive mask. The SFPU exps share the engine.
    double fpu = kFpuCyclesPerPhase * (work.qk_phases * scores * d + work.pv_phases * r * k * v +
                                       work.denominator_phases * scores);
    fpu += fused ? kFpuCyclesPerPhase * scores : scores * (kFpuReduceCycles + kFpuEltwiseCycles);
    fpu += fp32 ? 2 * kFpuEltwiseCycles * r * v : 0.0;
    fpu += masked ? kFpuEltwiseCycles * scores : 0.0;
    const double sfpu = kSfpuIterationsPerTile * work.exp_instructions * scores;
    const double subblocks = r * div_up(k_tiles, build.qk_width) + r * v / build.pv_width;
    const double math = fpu + sfpu + kDestHandoffCycles * subblocks;
    // Pack thread: it issues the exps, then packs per score tile P and the row sums (reduce path: S as well; FP32: S
    // and P, ACCURATE also S - m by L1 accumulation) and per row the FP32 O update.
    const double score_bytes = fp32 ? (policy.selection.recipe == Recipe::D ? 3 : 2) * state
                                    : 2 * kBf16Tile + (fused ? 0 : kBf16Tile);
    double pack =
        sfpu + (scores * score_bytes + r * v * kFp32Tile) / kPackBytesPerCycle + kDestHandoffCycles * subblocks;
    if (policy.selection.recipe == Recipe::E && fused) {
        pack += kFastPackSetups * (kBf16Tile / kPackBytesPerCycle) * (r * div_up(k_tiles, build.qk_width));
    }
    // Unpacker: one new operand tile per tile product (K, V), P once per PV subblock, S twice on the reduce path.
    double unpack_bytes = scores * d * kb + r * k * v * vb + r * k * (v / build.pv_width) * state;
    unpack_bytes += fused ? 0.0 : 2 * scores * state;
    unpack_bytes += masked ? scores * p.mask_page_bytes : 0.0;
    const double unpack = unpack_bytes / kUnpackBytesPerCycle;
    const double groups = div_up(rows, group_rows(p, rows));
    double syncs = kSyncCycles * syncs_per_group(policy, fused, vd) * groups;
    // BF16 reduce path: each QK subblock's S is packed and unpacked again for the row max (one round trip). Matches
    // the measured cost of losing the fused chunk: FAST 1.28x (LOW_PRECISION study), STANDARD 1.09x (one core).
    syncs += !fused && !fp32 ? kSyncCycles * r * div_up(k_tiles, build.qk_width) : 0.0;
    return std::max({math, pack, unpack}) + syncs;
}

// Per-Q-chunk compute outside the blocks: normalization (unpack O, scale by 1 / l, pack the output) and, with fused
// chunks, the first K chunk on the reduce path.
double q_chunk_compute(
    const RecipeBlockingProblem& p, uint32_t rows, uint32_t k_tiles, const RecipeBuild& build, bool dense, bool fused) {
    const uint32_t vd = dense && p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles;
    const double norm = static_cast<double>(rows) * vd *
                        (kFp32Tile / kUnpackBytesPerCycle + kFpuEltwiseCycles + kBf16Tile / kPackBytesPerCycle);
    const double first = fused ? block_compute(p, rows, k_tiles, build, dense, false, false) -
                                     block_compute(p, rows, k_tiles, build, dense, true, false)
                               : 0.0;
    return norm + first;
}

// A Q chunk's time: its compute against its K/V transfers into the core (`kv_noc`) and its DRAM time. Two K/V slots
// (BF16 recipes) overlap the next chunk's transfer with compute; one slot (FP32 recipes) serializes it.
double q_chunk_time(const PrecisionPolicy& policy, double compute, double kv_noc, double dram) {
    return policy.fp32_destination ? std::max(compute + kv_noc, dram) : std::max({compute, kv_noc, dram});
}
// Ring joint (ring_joint_sdpa_program_factory.cpp) adds to the B-E recipe layout: the lightweight
// mask (neginf plus up to two partial-tile masks), the scale scalar, the two 4 KiB state-transfer
// pages (CBs 17/18), the signal page and the derived-geometry mailbox. Partial masks are counted
// unconditionally, so this can only over-estimate.
constexpr uint64_t kRingRecipeExtraBytes = 3 * kBf16Tile + kBf16Tile + 2 * 4096 + 16 + 64;

// The dense / joint circular-buffer facts of a problem.
RecipeL1Context dense_l1_context(const RecipeBlockingProblem& p) {
    return {
        .mask_page_bytes = p.mask_page_bytes,
        .extra_bytes = p.extra_l1_bytes,
        .keyed = p.key_range,
        .vd_tiles = p.vd_tiles,
        .v_storage = p.v_storage};
}

std::vector<uint32_t> tile_range(uint32_t fixed, uint32_t lo, uint32_t hi) {
    if (fixed != 0) {
        return {fixed};
    }
    std::vector<uint32_t> values;
    for (uint32_t t = lo; t <= hi; ++t) {
        values.push_back(t);
    }
    return values;
}

}  // namespace

std::optional<std::string> recipe_geometry_rejection(uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
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

void validate_recipe_geometry(RecipeOp op, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim) {
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
    const auto rejection = recipe_geometry_rejection(q_chunk_size / kTile, k_chunk_size / kTile, head_dim / kTile);
    TT_FATAL(
        !rejection,
        "Named {} SDPA recipes do not support Q{}/K{}/D{}: {}",
        name,
        q_chunk_size,
        k_chunk_size,
        head_dim,
        rejection.value_or(""));
}

uint32_t recipe_mask_group_rows(const PrecisionPolicy& policy) { return policy.fp32_destination ? 1 : 2; }

RecipeL1Estimate recipe_l1_bytes(
    RecipeOp op,
    const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    const RecipeL1Context& context) {
    const auto rejection = recipe_geometry_rejection(q_tiles, k_tiles, d_tiles);
    TT_FATAL(!rejection, "Unsupported SDPA recipe geometry: {}", *rejection);
    const uint64_t q_slot = uint64_t{q_tiles} * d_tiles * kBf16Tile;
    // The factories' L1 fallback drops the fused chunks' CBs, except for an odd STANDARD Q chunk (recipe_drop_fused).
    auto droppable_fused = [&](uint32_t rows, const RecipeBuild& build) -> uint64_t {
        return rows % 2 != 0 && policy.selection.recipe == Recipe::B ? 0 : build.fused_cb_bytes;
    };
    switch (op) {
        case RecipeOp::Dense:
        case RecipeOp::Joint: {
            // Key ranges mask only their edge K chunks and keep the fused chunks; an attn_mask build has none.
            const bool masked = context.mask_page_bytes > 0 && !context.keyed;
            const uint32_t rows =
                recipe_compute_q_tiles(policy, q_tiles, k_tiles, context.mask_page_bytes > 0, context.keyed);
            const auto build = recipe_build(policy, rows, k_tiles, d_tiles, context.vd_tiles, context.v_storage);
            // attn_mask CB: two row groups when they fit, one otherwise (sdpa_recipe.cpp). A masked build has no
            // fused chunks' CBs; the minimum layout drops them in any case (check_recipe_l1_fit).
            const uint64_t mask_group = uint64_t{recipe_mask_group_rows(policy)} * k_tiles * context.mask_page_bytes;
            const uint64_t cbs = build.cb_bytes - (masked ? build.fused_cb_bytes : 0);
            return {
                cbs + 2 * mask_group + context.extra_bytes,
                build.cb_bytes - droppable_fused(rows, build) + mask_group + context.extra_bytes};
        }
        case RecipeOp::Ring:
        case RecipeOp::ExpRing: {
            // The ring factories size K and V for two entries each (the recipe layout's FP32 recipes have one).
            const auto build = recipe_build(policy, q_tiles, k_tiles, d_tiles);
            const uint32_t vd = context.vd_tiles != 0 ? context.vd_tiles : d_tiles;
            const uint64_t two_kv = 2ull * k_tiles * (d_tiles + vd) * kv_tile_bytes(policy.selection.kv_storage);
            const uint64_t layout = build.cb_bytes - build.kv_cb_bytes + two_kv;
            if (op == RecipeOp::Ring) {
                const uint64_t bytes = layout + kRingRecipeExtraBytes;
                // The factory's fallbacks: single-slot Q, then drop the fused chunks' CBs.
                return {bytes, bytes - droppable_fused(q_tiles, build) - q_slot};
            }
            // Exp ring: a single Q slot, plus the 64 B live-length mailbox of a device-tensor logical_n (counted
            // unconditionally); its fallbacks keep one K and one V entry, then drop the fused chunks' CBs.
            const uint64_t bytes = layout - q_slot + 64;
            return {bytes, bytes - two_kv / 2 - droppable_fused(q_tiles, build)};
        }
    }
    TT_THROW("Unknown SDPA recipe op");
}

bool recipe_program_fits(const RecipeBlockingProblem& p, uint32_t q_tiles, uint32_t k_tiles) {
    // A program's RISC images, runtime args, semaphores and CB configs share one 70656 B kernel config buffer per core.
    // Measured exclusions on one P150b; the per-case program sizes are in test_recipe_blocking.cpp (ProgramsFit*,
    // FeatureCombinations*). Exp ring programs are not covered.
    if (p.op == RecipeOp::Ring) {
        // MLA-width rings (D576 Q/K with a narrower V, the DeepSeek MLA prefill shape) sit within ~100 B of the
        // buffer from Q160 up: causal balanced ACCURATE with BFP8 K/V on a 2x2 P150b mesh measured 70672 B at
        // Q160/K192 and K448, 70688 at Q160/K256, 70720 at Q160/K384 (K160, K224, K288, K320 and K512 fit by a few
        // bytes) and 71184-71696 B at Q320; every Q32 and Q64 blocking fits.
        return !(p.vd_tiles != 0 && p.vd_tiles < p.d_tiles && p.d_tiles >= 16 && q_tiles >= 5);
    }
    if (p.op != RecipeOp::Dense && p.op != RecipeOp::Joint) {
        return true;
    }
    const Recipe recipe = p.policy.selection.recipe;
    const bool attn_mask = p.mask_page_bytes > 0 && !p.key_range;
    const bool k_tail = p.k_rows_unaligned || (p.k_rows + p.joint_k_rows) % (k_tiles * kTile) != 0;
    const bool feature = p.op == RecipeOp::Joint || p.attention_sink || k_tail;
    const uint32_t pv_tiles = p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles;
    const bool narrow_pv = pv_tiles % 2 != 0;
    const bool packed =
        p.policy.selection.kv_storage != KVStorage::BF16 || p.v_storage.value_or(KVStorage::BF16) != KVStorage::BF16;
    if (recipe == Recipe::E) {
        const bool odd = q_tiles % 2 != 0 && q_tiles >= 3 && !p.key_range;
        return !(odd && ((narrow_pv && feature) || (packed && p.attention_sink && (k_tail || attn_mask))));
    }
    if (recipe == Recipe::C) {
        return q_tiles < 5 || !((pv_tiles == 3 && (feature || p.mask_page_bytes > 0)) ||
                                (pv_tiles == 2 && p.key_range && p.attention_sink));
    }
    if (recipe != Recipe::B || attn_mask || recipe_qk_subblock_width(k_tiles, p.policy) < 2) {
        return true;
    }
    if (q_tiles % 2 != 0 && q_tiles >= 7 && (feature || narrow_pv)) {
        return false;
    }
    if (p.attention_sink && k_tail && !p.key_range && k_tiles == 8 && q_tiles >= 8 &&
        (narrow_pv || (packed && pv_tiles == 2))) {
        return false;
    }
    return !(narrow_pv && packed && (feature || (p.paged && p.sliding_window > 0)));
}

namespace {
// The K chunks [first, end) each Q chunk of a key-range call processes and how many of them are edge chunks (partly
// masked), mirroring RecipeKeyRange::chunks in dataflow/recipe_key_range.hpp: causal and sliding-window rows at
// q_offset + row.
std::vector<std::pair<uint32_t, uint32_t>> key_range_chunks(
    const RecipeBlockingProblem& p, uint32_t q_tiles, uint32_t k_tiles) {
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
    std::vector<std::pair<uint32_t, uint32_t>> chunks;
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
        chunks.emplace_back(end - first, (end - first) - (full_end - full_begin));
    }
    return chunks;
}

// The busiest core's time when all heads' Q chunks (each head with the same `job_costs`) are dealt to `cores` in snake
// order by cost (run_recipe_segments: recipe_snake_count).
double snake_makespan(const RecipeBlockingProblem& p, std::vector<double> job_costs, uint32_t cores) {
    // Every head has the same Q chunk costs: sorted entry s is job s / heads of the descending list.
    std::sort(job_costs.begin(), job_costs.end(), std::greater<>());
    const uint32_t heads = p.batch * p.q_heads;
    const uint64_t total = uint64_t{heads} * job_costs.size();
    std::vector<double> load(cores, 0.0);
    for (uint64_t s = 0; s < total; ++s) {
        const uint64_t round = s / cores, position = s % cores;
        load[round % 2 == 0 ? position : cores - 1 - position] += job_costs[s / heads];
    }
    return *std::max_element(load.begin(), load.end());
}

// `widen`: search chunks down to one tile (the fallback when nothing in the searched range fits L1).
std::vector<RecipeBlocking> blocking_candidates(const RecipeBlockingProblem& p, bool widen) {
    std::vector<RecipeBlocking> candidates;
    const uint32_t batch_heads = p.batch * p.q_heads;
    if (batch_heads == 0 || p.q_rows == 0 || p.k_rows == 0 || p.grid.x == 0 || p.grid.y == 0) {
        return candidates;
    }
    const auto& arch = arch_quantities(p);
    // A chunk longer than the sequence only adds padding; the default sizes stay in range.
    const bool dense = p.op == RecipeOp::Dense || p.op == RecipeOp::Joint;
    const uint32_t q_cap = std::min(kRecipeSearchMaxQTiles, std::max(div_up(p.q_rows + p.joint_q_rows, kTile), 10u));
    // Padded K blocks are costed, so short K picks short chunks. Fused FAST amortizes its per-chunk
    // work (saturation check, fold, PV pieces) over longer K chunks: up to K1024 when L1 allows.
    const uint32_t k_cap = p.policy.selection.recipe == Recipe::E ? 2 * kRecipeSearchMaxKTiles : kRecipeSearchMaxKTiles;
    // Few batch/heads with long K whose Q128 chunks cannot fill the grid (single-head VAE attention): Q chunks down to
    // 64 rows spread a head over more cores, with K chunks down to 128 rows (one head, ACCURATE, traced: Wan VAE 480p
    // D384 S6240 Q64/K128 1.08 ms, Q64/K256 1.18, the Q128/K256 choice 1.59; SD 1.4 VAE D512 S4096 Q64/K128 0.89,
    // Q64/K256 1.06, Q128/K128 1.55).
    const bool unfilled = dense && p.k_rows + p.joint_k_rows >= 2048 &&
                          uint64_t{batch_heads} * div_up(p.q_rows + p.joint_q_rows, kRecipeSearchMinQTiles * kTile) <
                              uint64_t{p.grid.x} * p.grid.y;
    const uint32_t q_min = widen ? 1u : unfilled ? kRecipeSearchMinQTiles / 2 : kRecipeSearchMinQTiles;
    const uint32_t q_floor = std::min(q_min, div_up(p.q_rows + p.joint_q_rows, kTile));
    // A sliding window bounds each Q chunk's keys, so K chunks down to 128 rows waste fewer masked keys.
    const uint32_t k_min =
        (p.key_range && p.sliding_window > 0) || unfilled ? kRecipeSearchMinKTiles / 2 : kRecipeSearchMinKTiles;
    const uint32_t k_floor = widen ? 1u : std::min(k_min, div_up(p.k_rows + p.joint_k_rows, kTile));
    const auto q_range = tile_range(p.fixed_q_tiles, q_floor, q_cap);
    const auto k_range = tile_range(p.fixed_k_tiles, k_floor, k_cap);
    const uint32_t kv_page = kv_tile_bytes(p.policy.selection.kv_storage);
    const uint32_t vd = dense && p.vd_tiles != 0 ? p.vd_tiles : p.d_tiles;
    for (const uint32_t qt : q_range) {
        if (p.q_chunk_divides != 0 && p.q_chunk_divides % (qt * kTile) != 0) {
            continue;
        }
        bool any_fit = false, size_excluded = false;
        for (const uint32_t kt : k_range) {
            if (!recipe_geometry_supported(qt, kt, p.d_tiles)) {
                continue;
            }
            if ((dense || p.op == RecipeOp::Ring) && !recipe_program_fits(p, qt, kt)) {
                size_excluded = true;
                continue;
            }
            // Ring / exp ring round STANDARD's odd Q chunk up to the next even one (sdpa.cpp); the even
            // candidate is costed on its own.
            if (!dense && qt % 2 != 0 && recipe_compute_q_tiles(p.policy, qt, kt) != qt) {
                continue;
            }
            // Dense/joint L1 grows with Q and K: stop at the first K that does not fit.
            if (dense && recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, dense_l1_context(p)).minimum > p.l1_bytes) {
                break;
            }
            any_fit = true;
            const uint32_t q_chunk = qt * kTile;
            const uint32_t k_chunk = kt * kTile;
            const RecipeBuild build =
                recipe_build(p.policy, qt, kt, p.d_tiles, dense ? p.vd_tiles : 0, dense ? p.v_storage : std::nullopt);
            const bool attn_mask = dense && p.mask_page_bytes > 0 && !p.key_range;
            // Dense/joint unfused STANDARD, and paired recipes with a key range, compute an odd Q chunk with one
            // padding row (recipe_compute_q_tiles).
            const uint32_t rows = dense ? recipe_compute_q_tiles(p.policy, qt, kt, p.key_range, p.key_range) : qt;
            const KvChunk kv = kv_chunk(p, kt, dense);
            const double q_bytes = static_cast<double>(qt) * p.d_tiles * kBf16Tile;
            const double o_bytes = static_cast<double>(qt) * vd * kBf16Tile;
            // Pipeline fill and drain of a core running Q chunks: every core reads its first Q chunk and K/V block
            // before computing (all at once, through the shared DRAM) and writes its last output chunk after.
            const uint32_t first_k_tiles = std::min(kt, std::max(1u, div_up(p.k_rows, kTile)));
            const KvChunk first_kv = kv_chunk(p, first_k_tiles, dense);
            auto fill_drain = [&](double used, double readers) {
                const double fill = std::max(
                    (used * q_bytes + readers * first_kv.bytes) / arch.dram_bytes_per_cycle,
                    read_cycles(arch, first_kv.tiles, kv_page, kChainReadBarrier));
                return fill + used * o_bytes / arch.dram_bytes_per_cycle;
            };
            // `work(fused)`: the busiest core's Q chunks, whether its K chunks run fused (the factories drop the fused
            // chunks' CBs when the preferred layout does not fit L1; every chunk then runs the reduce path).
            auto admit = [&](uint32_t jobs,
                             CoreCoord grid,
                             const RecipeL1Context& context,
                             double used,
                             double readers,
                             const std::function<double(bool)>& work) {
                const auto l1 = recipe_l1_bytes(p.op, p.policy, qt, kt, p.d_tiles, context);
                if (l1.minimum > p.l1_bytes) {
                    return;
                }
                // The ring factories give up Q prefetch (ring) or a K/V entry (exp ring) before the fused chunks' CBs.
                // (An odd STANDARD Q chunk keeps them in any case: recipe_drop_fused.)
                const bool keeps_fused = qt % 2 != 0 && p.policy.selection.recipe == Recipe::B;
                const uint64_t fused_layout =
                    dense ? l1.preferred : l1.minimum + (keeps_fused ? 0 : build.fused_cb_bytes);
                const bool fused = build.fused && !attn_mask && fused_layout <= p.l1_bytes;
                candidates.push_back({q_chunk, k_chunk, grid, work(fused) + fill_drain(used, readers), jobs, l1});
            };
            auto block = [&](bool fused) { return block_compute(p, rows, kt, build, dense, fused, attn_mask); };
            auto q_chunk_work = [&](bool fused) { return q_chunk_compute(p, rows, kt, build, dense, fused); };
            switch (p.op) {
                case RecipeOp::Dense:
                case RecipeOp::Joint: {
                    // run_recipe_segments: one KV-forwarding chain per batch/head, jobs split evenly; with more
                    // batch/heads than cores, every head's Q chunks split evenly over the grid.
                    const uint32_t cores = p.grid.x * p.grid.y;
                    const uint32_t jobs_per_head = div_up(p.q_rows + p.joint_q_rows, q_chunk);
                    const uint32_t chain = std::min(jobs_per_head, cores / batch_heads);
                    if (p.key_range) {
                        // run_recipe_segments: all heads' Q chunks dealt over the grid in snake order by cost. No
                        // chains: every core reads its own K/V, except that K/V sharing partners split a full chunk's
                        // read (one forwards it; the FP32 recipes share no window ranges). Unshared reads wait on a
                        // barrier every `barrier` tiles (sdpa_recipe_program.cpp: read_barrier_tiles).
                        const uint32_t total_jobs = batch_heads * jobs_per_head;
                        const uint32_t used = std::min(cores, total_jobs);
                        const uint32_t jobs = div_up(total_jobs, used);
                        const double share = !p.policy.fp32_destination || p.sliding_window == 0 ? 0.5 : 1.0;
                        const uint32_t barrier = std::clamp<uint32_t>((512 / used) * (1024 + 128) / kv_page, 1, 16);
                        const double shared_read = read_cycles(arch, kv.tiles, kv_page, share < 1.0 ? 16 : barrier);
                        const double edge_read = read_cycles(arch, kv.tiles, kv_page, barrier);
                        // Windowed segments (unknown on the host) cost every K chunk.
                        const auto chunks =
                            p.causal || p.sliding_window > 0
                                ? key_range_chunks(p, qt, kt)
                                : std::vector<std::pair<uint32_t, uint32_t>>(
                                      jobs_per_head, {div_up(p.k_rows, k_chunk), 0u});
                        auto work = [&](bool fused) {
                            const double full_block = block(fused);
                            // An edge chunk takes the additive mask on the reduce path.
                            const double edge_block = block_compute(p, rows, kt, build, dense, false, true);
                            const double per_q = q_chunk_work(fused);
                            std::vector<double> job_costs;
                            job_costs.reserve(chunks.size());
                            for (const auto& [n, edges] : chunks) {
                                const double full = n - edges;
                                const double compute = full * full_block + edges * edge_block + per_q;
                                const double dram = std::max(
                                    used * (full * share * kv.bytes + edges * kv.bytes + q_bytes + o_bytes) /
                                        arch.dram_bytes_per_cycle,
                                    full * share * shared_read + edges * edge_read);
                                job_costs.push_back(
                                    q_chunk_time(p.policy, compute, n * kv.bytes / arch.noc_bytes_per_cycle, dram));
                            }
                            return snake_makespan(p, std::move(job_costs), used);
                        };
                        admit(jobs, p.grid, dense_l1_context(p), used, used, work);
                        break;
                    }
                    const uint32_t k_blocks = div_up(p.k_rows + p.joint_k_rows, k_chunk);
                    // Chains: the chain head reads each K/V chunk from DRAM and the members forward it; with more
                    // batch/heads than cores every core reads its own Q chunks' K/V.
                    const double used = chain == 0 ? cores : static_cast<double>(batch_heads) * chain;
                    const double readers = chain == 0 ? cores : batch_heads;
                    const uint32_t jobs =
                        chain == 0 ? div_up(batch_heads * jobs_per_head, cores) : div_up(jobs_per_head, chain);
                    const double mask_bytes = attn_mask ? static_cast<double>(rows) * kt * p.mask_page_bytes : 0.0;
                    // A chain of two or more cores streams each K/V chunk at the chain rate.
                    const double stream = chain >= 2 ? k_blocks * kv.bytes / arch.chain_stream_bytes_per_cycle : 0.0;
                    auto work = [&](bool fused) {
                        const double compute = k_blocks * block(fused) + q_chunk_work(fused);
                        const double dram = std::max(
                            {(readers * k_blocks * kv.bytes + used * (k_blocks * mask_bytes + q_bytes + o_bytes)) /
                                 arch.dram_bytes_per_cycle,
                             k_blocks * read_cycles(arch, kv.tiles, kv_page, kChainReadBarrier),
                             stream});
                        const double kv_noc = k_blocks * kv.bytes / arch.noc_bytes_per_cycle;
                        return jobs * q_chunk_time(p.policy, compute, kv_noc, dram);
                    };
                    admit(jobs, p.grid, dense_l1_context(p), used, readers, work);
                    break;
                }
                case RecipeOp::Ring:
                case RecipeOp::ExpRing: {
                    // The ring factories deliver every ring step's K/V shard to each worker over the NoC (DRAM traffic
                    // depends on their K/V distribution and is not modeled).
                    auto ring_work = [&](uint32_t jobs, uint32_t k_blocks, double stream) {
                        return [&, jobs, k_blocks, stream](bool fused) {
                            const double compute = k_blocks * block(fused) + q_chunk_work(fused);
                            const double kv_noc = k_blocks * kv.bytes / arch.noc_bytes_per_cycle;
                            return jobs * q_chunk_time(p.policy, compute, kv_noc, stream);
                        };
                    };
                    const uint32_t k_blocks = p.ring_size * div_up(p.k_rows, k_chunk) + div_up(p.joint_k_rows, k_chunk);
                    if (p.op == RecipeOp::Ring) {
                        // ring_joint_sdpa_program_factory.cpp: all Q chunks of all heads split evenly over
                        // the worker grid; every ring iteration streams the local KV shard, the joint KV once.
                        const uint32_t cores = p.grid.x * p.grid.y;
                        const uint32_t q_chunks = div_up(p.q_rows, q_chunk) + div_up(p.joint_q_rows, q_chunk);
                        const uint32_t jobs = div_up(batch_heads * q_chunks, cores);
                        // A head whose Q chunks span two or more cores shares its K/V through a store-and-forward
                        // chain (ring_joint_kv_chains.hpp), which streams every K block once per Q chunk at the chain
                        // rate, as the dense chains do.
                        const bool chained = q_chunks > jobs;
                        const double stream = chained ? k_blocks * kv.bytes / arch.chain_stream_bytes_per_cycle : 0.0;
                        // Only a chain's first core reads K/V from DRAM (the pipeline fill).
                        const double readers = chained ? std::min<double>(cores, batch_heads) : cores;
                        admit(
                            jobs,
                            p.grid,
                            RecipeL1Context{.vd_tiles = p.vd_tiles},
                            cores,
                            readers,
                            ring_work(jobs, k_blocks, stream));
                        break;
                    }
                    // exp_ring_joint_sdpa_device_operation.cpp: a head's Q chunks fill one core row
                    // exactly (num_q_chunks == cols * segs_per_head); rows walk head-segments as serial
                    // passes. The joint Q length must divide by the Q chunk.
                    if (p.joint_q_rows % q_chunk != 0) {
                        break;
                    }
                    if (p.grid.x < 2) {
                        break;
                    }
                    const uint32_t max_cols = p.grid.x - 1;  // the last column runs the fabric MUX
                    const uint32_t rows_of_cores = p.grid.y;
                    const uint32_t q_chunks = div_up(p.q_rows, q_chunk) + p.joint_q_rows / q_chunk;
                    // An explicit Q chunk keeps the caller's grid; otherwise the op may narrow it, down
                    // to the factory's minimum of 3 SDPA columns (one pure SDPA plus two MUX writers).
                    const uint32_t min_cols = p.fixed_q_tiles != 0 ? max_cols : 3;
                    for (uint32_t cols = max_cols; cols >= min_cols; --cols) {
                        if (q_chunks % cols != 0) {
                            continue;
                        }
                        const uint32_t segs = q_chunks / cols;
                        const uint32_t segments = batch_heads * segs;
                        if (segments < rows_of_cores) {
                            continue;
                        }
                        const uint32_t passes = div_up(segments, rows_of_cores);
                        const CoreCoord grid(cols + 1, p.grid.y);
                        const double workers = static_cast<double>(cols) * rows_of_cores;
                        admit(passes, grid, RecipeL1Context{}, workers, workers, ring_work(passes, k_blocks, 0.0));
                    }
                    break;
                }
            }
        }
        if (dense && !any_fit && !size_excluded && p.fixed_q_tiles == 0) {
            break;  // nothing fits L1 at this Q, so no larger Q fits either (FAST excludes odd Q chunks by size only)
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
}  // namespace

std::vector<RecipeBlocking> recipe_blocking_candidates(const RecipeBlockingProblem& p) {
    auto candidates = blocking_candidates(p, false);
    if (candidates.empty() && (p.fixed_q_tiles == 0 || p.fixed_k_tiles == 0)) {
        // Large head dims (D512: Gemma-4 global layers, SD VAEs) fit L1 only below the searched range (ACCURATE D512:
        // Q128/K128, STANDARD Q64/K64): search down to one-tile chunks, so a fitting geometry is always found.
        candidates = blocking_candidates(p, true);
    }
    return candidates;
}

std::optional<RecipeBlocking> choose_recipe_blocking(const RecipeBlockingProblem& problem) {
    static std::mutex mutex;
    static std::map<RecipeBlockingProblem, std::optional<RecipeBlocking>> memo;
    {
        std::lock_guard lock(mutex);
        if (auto it = memo.find(problem); it != memo.end()) {
            return it->second;
        }
    }
    const auto candidates = recipe_blocking_candidates(problem);
    std::optional<RecipeBlocking> choice;
    if (!candidates.empty()) {
        // Candidates within kTieTolerance of the cheapest are ties: the default Q256/K512 (the most-measured blocking)
        // when among them, else the larger K chunk (fewer of the per-chunk handshakes the model leaves out), then the
        // larger Q chunk, then the cheaper.
        const double limit = candidates.front().cost * (1.0 + kTieTolerance);
        for (const auto& candidate : candidates) {
            if (candidate.cost > limit) {
                break;
            }
            if (candidate.q_chunk_size == kDefaultQTiles * kTile && candidate.k_chunk_size == kDefaultKTiles * kTile) {
                choice = candidate;
                break;
            }
            if (!choice || candidate.k_chunk_size > choice->k_chunk_size ||
                (candidate.k_chunk_size == choice->k_chunk_size && candidate.q_chunk_size > choice->q_chunk_size)) {
                choice = candidate;
            }
        }
    }
    std::lock_guard lock(mutex);
    if (memo.size() > 4096) {
        memo.clear();
    }
    memo.emplace(problem, choice);
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
    problem.fixed_q_tiles = fixed_tiles(config.q_chunk_size);
    problem.fixed_k_tiles = fixed_tiles(config.k_chunk_size);
    problem.wormhole = q.device()->arch() == tt::ARCH::WORMHOLE_B0;
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
    const std::optional<SDPAProgramConfig>& program_config,
    const Tensor* attn_mask,
    uint64_t reserved_l1_bytes,
    const RecipeKeyRange* key_range,
    const RecipeDenseOptions* options,
    bool chunks_are_hints,
    std::optional<KVStorage> v_storage) {
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
    if (v_storage != policy.selection.kv_storage) {
        problem.v_storage = v_storage;
    }
    problem.attention_sink = options && options->attention_sink.has_value();
    problem.k_rows_unaligned = (key_range ? recipe_k_rows(k, *key_range) : k.logical_shape()[2]) % kTile != 0 ||
                               (joint_k && joint_k->logical_shape()[2] % kTile != 0);
    const uint64_t free_l1 = recipe_free_l1(*device);
    problem.l1_bytes = free_l1 > reserved_l1_bytes ? free_l1 - reserved_l1_bytes : 0;
    problem.mask_page_bytes = attn_mask ? attn_mask->buffer()->page_size() : 0;
    if (key_range && key_range->active()) {
        problem.mask_page_bytes = 576;  // BFP4 mask tiles
        problem.extra_l1_bytes = recipe_key_range_extra_bytes(*key_range);
        problem.key_range = true;
        problem.causal = key_range->causal;
        problem.paged = key_range->page_table.has_value();
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
    if (chunks_are_hints) {
        // Routed calls choose their chunks (ring-distributed on one P150, 8/1 heads D128 BFP8, 131072 rows, STANDARD:
        // the caller's Q64/K64 143 ms, the choice 43 ms; Q256/K512 43 ms).
        problem.fixed_q_tiles = 0;
        problem.fixed_k_tiles = 0;
        config.q_chunk_size = 0;
        config.k_chunk_size = 0;
    }
    if (key_range && key_range->q_slab_rows) {
        // Ring-distributed SDPA: two Q slabs, each of whole Q chunks, on the caller's grid.
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
    }
    const auto choice = invalid_fixed(config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(config, choice, problem, op_name);
}

SDPAProgramConfig resolve_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t joint_q_rows,
    uint32_t joint_k_rows,
    uint32_t ring_size,
    uint32_t v_head_dim,
    uint32_t q_chunk_divides,
    const SDPAProgramConfig& program_config) {
    if (!recipe_blocking_requested(program_config)) {
        return program_config;
    }
    auto problem = base_problem(RecipeOp::Ring, policy, q, program_config);
    problem.vd_tiles = v_head_dim != 0 && v_head_dim != q.logical_shape()[3] ? div_up(v_head_dim, kTile) : 0;
    problem.q_rows = q_rows;
    problem.k_rows = k_rows;
    problem.joint_q_rows = joint_q_rows;
    problem.joint_k_rows = joint_k_rows;
    problem.ring_size = ring_size;
    problem.q_chunk_divides = q_chunk_divides;
    problem.l1_bytes = recipe_free_l1(*q.device());
    const auto choice = invalid_fixed(program_config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(program_config, choice, problem, "ring");
}

SDPAProgramConfig resolve_exp_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t joint_rows,
    uint32_t ring_size,
    const SDPAProgramConfig& program_config) {
    if (!recipe_blocking_requested(program_config)) {
        return program_config;
    }
    auto problem = base_problem(RecipeOp::ExpRing, policy, q, program_config);
    problem.q_rows = q_rows;
    problem.k_rows = k_rows;
    problem.joint_q_rows = joint_rows;
    problem.joint_k_rows = joint_rows;
    problem.ring_size = ring_size;
    problem.l1_bytes = recipe_free_l1(*q.device());
    const auto choice = invalid_fixed(program_config) ? std::nullopt : choose_recipe_blocking(problem);
    return apply_choice(program_config, choice, problem, "exp ring");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
