// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_config.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Shared hardware/layout constants (bank count, L1 budget, tile bytes, core-count window) live in
// regime_a_matmul_plan.hpp — the SINGLE source of truth — and are reached here via the `plan::` alias.
using plan::kL1BudgetBytes;
using plan::kNumBanks;
using plan::kTileBytesBf16;
using plan::kTileBytesFp32;

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t rup(uint32_t x, uint32_t y) { return cdiv(x, y) * y; }

// ---- Auto-selector (ported from tools/mm_sweep/picker_{table,cost-model}.py) ----
// Cost-model params (grid-search best on the 3262-config oracle: geomean 96.8%).
constexpr uint32_t kCsat = 24, kAcap = 6, kKbcap = 2;
constexpr double kKk = 0.5, kAa = 2.0, kOvl = 1.0, kStart = 0.0, kWst = 0.5;
// v3 M-split fallback (trained on the Mt<=8 campaign, tools/mm_sweep/picker_v3.py): the deployed Sm=1
// ranking is the ANCHOR; an Sm>1 candidate is chosen only for NARROW-N shapes (Nband<=kNbandMax, where
// N-split cannot supply parallelism) when its reduction-aware cost beats the anchor's by kMSplitMargin.
// kRk penalises split-K reduction (rk*(Pk-1)*out-tiles/core). Zero-regression on all 60 campaign shapes.
constexpr double kRk = 0.8, kMSplitMargin = 0.03;
constexpr uint32_t kNbandMax = 2u;

// Lightweight geometry + feasibility (mirrors the picker cost-model plan()). Returns false if infeasible.
struct PickGeo {
    uint32_t cores, Ktl, Mblk, Nown, Nbpc;
    double wasteK, wasteN;
};
bool pick_plan(
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt,
    uint32_t Ns,
    uint32_t Pk,
    uint32_t Sm,
    uint32_t kb,
    uint32_t nsb,
    PickGeo& g) {
    // Share the planner's bank-interval feasibility so config=None never selects a shape build_plan()
    // will later reject (picker/planner parity). This constraint is a function of Nt only.
    if (!plan::nt_width_shard_feasible(Nt)) {
        return false;
    }
    g.cores = kNumBanks * Pk * Ns * Sm;
    if (g.cores < plan::kMinCores || g.cores > plan::kMaxCores) {
        return false;
    }
    g.Ktl = rup(cdiv(Kt, Pk), kb * kNumBanks);
    g.wasteK = static_cast<double>(Pk * g.Ktl) / Kt - 1.0;
    if (g.wasteK > 0.20) {
        return false;
    }
    g.Mblk = cdiv(Mt, Sm);
    const uint32_t Nband = cdiv(Nt, kNumBanks);
    g.Nown = cdiv(Nband, Ns);
    if (nsb > g.Nown) {
        return false;
    }
    g.Nbpc = cdiv(g.Nown, nsb);
    g.wasteN = static_cast<double>(kNumBanks * Ns * g.Nbpc * nsb) / Nt - 1.0;
    if (g.wasteN > 0.20) {
        return false;
    }
    const uint32_t cb0 = g.Ktl * g.Mblk * kTileBytesBf16, cb1 = 4u * kb * nsb * kTileBytesBf16,
                   cb2 = 2u * g.Mblk * nsb * kTileBytesBf16, cb3 = g.Mblk * nsb * kTileBytesFp32,
                   cb7 = 2u * g.Mblk * nsb * kTileBytesBf16;
    return (cb0 + cb1 + cb2 + cb3 + cb7) <= kL1BudgetBytes;
}

double pick_cost(uint32_t Kt, uint32_t Nt, uint32_t kb, uint32_t nsb, const PickGeo& g) {
    const double readT = static_cast<double>(Kt) * Nt / std::min(g.cores, kCsat);
    const double comp_pc = static_cast<double>(g.Mblk) * g.Nown * g.Ktl;
    const double area = std::min<double>(static_cast<double>(g.Mblk) * nsb, kAcap);
    const double kbe = std::min(kb, kKbcap);
    const double compT = comp_pc / ((kbe / (kbe + kKk)) * (area / (area + kAa)));
    const double ovlT = kOvl * comp_pc / g.Nbpc;
    const double base = std::max(readT, compT) + ovlT + kStart * g.Ktl;
    return base * (1.0 + kWst * (g.wasteK + g.wasteN));
}

// v3 reduction-aware cost: deployed cost + split-K reduction penalty (rk*(Pk-1)*output-tiles-per-core).
// Used ONLY for the narrow-N Sm>1 hysteresis so the Sm=1 ranking stays byte-identical to the deployed model.
double pick_cost_v3(uint32_t Kt, uint32_t Nt, uint32_t Pk, uint32_t kb, uint32_t nsb, const PickGeo& g) {
    const double reduce = kRk * (Pk > 1u ? static_cast<double>(Pk - 1u) : 0.0) * g.Mblk * g.Nown;
    return pick_cost(Kt, Nt, kb, nsb, g) + reduce;
}

}  // namespace

RegimeAMatmulConfig auto_select_config(uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    // Oracle lookup table (100% on the 20 FLUX/LTX production shapes), keyed by TILE dims (Mt,Kt,Nt).
    // value = {k_slices(Pk), n_slices(Ns), m_slices(Sm), k_block_tiles(kb), n_subblock_tiles(nsb)}.
    static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, RegimeAMatmulConfig> kTable = {
        {{1, 64, 16}, {4, 2, 1, 2, 1}},
        {{1, 64, 48}, {2, 2, 1, 4, 3}},
        {{1, 192, 48}, {6, 1, 1, 4, 2}},
        {{1, 64, 64}, {2, 2, 1, 4, 4}},
        {{1, 192, 72}, {3, 1, 1, 4, 5}},  // 32x6144x2304: LTX/FLUX campaign -3.8% (Pk3/kb4/nsb5 vs Pk4/kb2/nsb9)
        {{1, 192, 96}, {3, 1, 1, 4, 6}},
        {{1, 8, 192}, {1, 3, 1, 1, 8}},
        {{1, 192, 192}, {6, 1, 1, 4, 2}},
        {{1, 192, 288}, {3, 1, 1, 4, 6}},
        {{2, 192, 48}, {3, 1, 1, 8, 2}},  // 64x6144x1536: LTX/FLUX campaign -3.5% (Pk3/kb8 vs Pk12/kb2)
        {{2, 480, 48}, {6, 1, 1, 2, 3}},  // 64x15360x1536: LTX/FLUX campaign -2.3% (Pk6/kb2 vs Pk12/kb1)
        {{2, 192, 144}, {6, 1, 1, 4, 2}},
        {{2, 144, 192}, {3, 2, 1, 2, 3}},  // 64x4608x6144: LTX/FLUX campaign sweep winner, -2.8% vs {6,1,1,1,8}
        {{2, 192, 288}, {6, 1, 1, 4, 2}},
        {{4, 192, 24}, {12, 1, 1, 2, 1}},
        // (128,15360,768): ring-order corpus re-sweep found (Pk6,kb2,nsb3) a stable +6.5% over the old
        // (Pk12,kb1,nsb3) under the current pipelined-drain + pareto-ring stack (PCC 0.99999 fresh+cached).
        {{4, 480, 24}, {6, 1, 1, 2, 3}},
        {{4, 192, 72}, {12, 1, 1, 2, 1}},
        {{4, 192, 144}, {12, 1, 1, 2, 1}},
        {{4, 72, 192}, {3, 2, 1, 1, 6}},
        {{16, 192, 48}, {12, 1, 1, 2, 1}},
        // Mt=8 low-AI shape: the cost-model fallback picks N-split (Ns2) which is ~15% slower here than
        // M-split. Exhaustive op-side sweep (788 configs) winner; overhead/reduction-tail-bound at ~37%.
        // Mt<=8 re-baseline campaign (2026-07, tools/mm_sweep/picker_v3.py + regime_a_campaign): measured
        // winners for the M-scaling shapes, +3..+32% vs the old fallback, all zero-regression (stability /
        // exhaustive-expand / validate confirmed; single-run entries re-validated by the gated corpus re-run).
        {{8, 64, 32}, {4, 1, 2, 2, 4}},    // 256x2048x1024 +5% (was {4,1,2,2,2})
        {{8, 480, 48}, {6, 1, 2, 2, 6}},   // 256x15360x1536 +32%
        {{8, 64, 16}, {4, 1, 3, 2, 2}},    // 256x2048x512 +24%
        {{1, 480, 24}, {6, 1, 1, 2, 3}},   // 32x15360x768 +24%
        {{8, 72, 192}, {3, 4, 1, 1, 3}},   // 256x2304x6144 +21%
        {{4, 64, 16}, {4, 1, 2, 2, 2}},    // 128x2048x512 +21%
        {{4, 480, 48}, {12, 1, 1, 1, 3}},  // 128x15360x1536 +21%
        {{2, 64, 16}, {4, 2, 1, 2, 1}},    // 64x2048x512 +21%
        {{8, 64, 48}, {4, 1, 3, 2, 3}},    // 256x2048x1536 +19%
        {{1, 64, 32}, {2, 4, 1, 4, 1}},    // 32x2048x1024 +19%
        {{2, 64, 32}, {4, 2, 1, 2, 2}},    // 64x2048x1024 +14%
        {{8, 480, 24}, {6, 1, 2, 2, 3}},   // 256x15360x768 +10%
        {{8, 64, 64}, {4, 1, 3, 2, 4}},    // 256x2048x2048 +9%
        {{4, 64, 32}, {4, 1, 2, 2, 4}},    // 128x2048x1024 +9%
        {{8, 192, 48}, {6, 1, 2, 4, 2}},   // 256x6144x1536 +8%
        {{1, 72, 192}, {3, 2, 1, 1, 6}},   // 32x2304x6144 +7%
        {{2, 64, 64}, {2, 3, 1, 2, 3}},    // 64x2048x2048 +5%
        {{1, 480, 48}, {6, 1, 1, 2, 3}},   // 32x15360x1536 +5%
        {{8, 192, 192}, {6, 1, 2, 4, 2}},  // 256x6144x6144 +4%
        {{4, 64, 64}, {4, 3, 1, 2, 3}},    // 128x2048x2048 +4%
        {{2, 480, 24}, {10, 1, 1, 2, 3}},  // 64x15360x768 +4%
        {{1, 192, 24}, {6, 1, 1, 2, 3}},   // 32x6144x768 +4%
        {{8, 192, 144}, {6, 1, 2, 4, 2}},  // 256x6144x4608 +3%
    };
    if (auto it = kTable.find({Mt, Kt, Nt}); it != kTable.end()) {
        return it->second;
    }

    // Cost-model fallback. Step 1: the deployed Sm=1 ANCHOR (min deployed cost) -- unchanged behaviour.
    RegimeAMatmulConfig anchor{};
    double anchor_cost = std::numeric_limits<double>::infinity();
    PickGeo anchor_g{};
    const uint32_t Nband = cdiv(Nt, kNumBanks);
    for (uint32_t Pk = 1; Pk <= 12u; ++Pk) {
        for (uint32_t Ns = 1; Ns <= 6u; ++Ns) {
            const uint32_t Nown = cdiv(Nband, Ns);
            for (uint32_t kb : {1u, 2u, 4u, 8u}) {
                for (uint32_t nsb = 1; nsb <= Nown; ++nsb) {
                    PickGeo g{};
                    if (!pick_plan(Mt, Kt, Nt, Ns, Pk, 1u, kb, nsb, g)) {
                        continue;
                    }
                    const double c = pick_cost(Kt, Nt, kb, nsb, g);
                    if (c < anchor_cost) {
                        anchor_cost = c;
                        anchor_g = g;
                        anchor = RegimeAMatmulConfig{
                            .k_slices = Pk,
                            .n_slices = Ns,
                            .m_slices = 1u,
                            .k_block_tiles = kb,
                            .n_subblock_tiles = nsb};
                    }
                }
            }
        }
    }
    TT_FATAL(
        anchor_cost != std::numeric_limits<double>::infinity(),
        "regime_a_matmul auto-select found no feasible config for Mt={} Kt={} Nt={}",
        Mt,
        Kt,
        Nt);

    // Step 2: NARROW-N M-split hysteresis. Only where N-split cannot supply parallelism (Nband<=kNbandMax)
    // do we consider Sm>1, and only adopt it when its reduction-aware cost beats the anchor's by the margin.
    // Otherwise the anchor (deployed pick) is returned -> zero regression by construction.
    if (Nband > kNbandMax || Mt < 2u) {
        return anchor;
    }
    RegimeAMatmulConfig bestG{};
    double bestG_cost = std::numeric_limits<double>::infinity();
    for (uint32_t Pk = 1; Pk <= 12u; ++Pk) {
        for (uint32_t Ns = 1; Ns <= 6u; ++Ns) {
            const uint32_t Nown = cdiv(Nband, Ns);
            for (uint32_t Sm = 2; Sm <= Mt; ++Sm) {
                for (uint32_t kb : {1u, 2u, 4u, 8u}) {
                    for (uint32_t nsb = 1; nsb <= Nown; ++nsb) {
                        PickGeo g{};
                        if (!pick_plan(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb, g)) {
                            continue;
                        }
                        const double c = pick_cost_v3(Kt, Nt, Pk, kb, nsb, g);
                        if (c < bestG_cost) {
                            bestG_cost = c;
                            bestG = RegimeAMatmulConfig{
                                .k_slices = Pk,
                                .n_slices = Ns,
                                .m_slices = Sm,
                                .k_block_tiles = kb,
                                .n_subblock_tiles = nsb};
                        }
                    }
                }
            }
        }
    }
    const double anchor_cost_v3 =
        pick_cost_v3(Kt, Nt, anchor.k_slices, anchor.k_block_tiles, anchor.n_subblock_tiles, anchor_g);
    if (bestG_cost < std::numeric_limits<double>::infinity() && bestG_cost < anchor_cost_v3 * (1.0 - kMSplitMargin)) {
        return bestG;
    }
    return anchor;
}

plan::PlanResult make_and_build_plan(
    IDevice* device, const Tensor& in0, const Tensor& in1, const std::optional<RegimeAMatmulConfig>& cfg_opt) {
    // Tile counts from logical shapes (tile = 32).
    const auto& a_shape = in0.logical_shape();
    const auto& w_shape = in1.logical_shape();
    const uint32_t Mt = cdiv(static_cast<uint32_t>(a_shape[-2]), TILE_HEIGHT);
    const uint32_t Kt = cdiv(static_cast<uint32_t>(a_shape[-1]), TILE_WIDTH);
    const uint32_t Nt = cdiv(static_cast<uint32_t>(w_shape[-1]), TILE_WIDTH);

    // config=None -> auto-select (deterministic in (Mt,Kt,Nt), so program-cache-safe: the cache key is
    // (nullopt config + tensor shapes) and the same shapes always resolve to the same config).
    const RegimeAMatmulConfig cfg = cfg_opt.value_or(auto_select_config(Mt, Kt, Nt));

    const CoreCoord grid = device->compute_with_storage_grid_size();

    auto to_plan_xy = [](const std::vector<CoreCoord>& src) {
        std::vector<plan::PlanXY> out;
        out.reserve(src.size());
        for (const auto& c : src) {
            out.push_back(plan::PlanXY{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)});
        }
        return out;
    };
    const auto opt0 = to_plan_xy(device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0));
    const auto opt1 = to_plan_xy(device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1));

    plan::PlanInputs in;
    in.Mt = Mt;
    in.Kt = Kt;
    in.Nt = Nt;
    in.cfg = plan::RegimeAConfig{
        .k_slices = cfg.k_slices,
        .n_slices = cfg.n_slices,
        .m_slices = cfg.m_slices,
        .k_block_tiles = cfg.k_block_tiles,
        .n_subblock_tiles = cfg.n_subblock_tiles};
    in.grid_x = static_cast<uint32_t>(grid.x);
    in.grid_y = static_cast<uint32_t>(grid.y);
    in.opt0 = opt0;
    in.opt1 = opt1;
    in.holes = {};  // v1: no explicit grid holes; find_near just walks to the next free logical core.
    // BH usable L1 ~1440 KB; matches the validated prototype/sweep budget used by the picker.
    in.l1_budget_bytes = kL1BudgetBytes;
    in.tb = kTileBytesBf16;  // bf16 tile bytes
    in.tf = kTileBytesFp32;  // fp32 tile bytes

    return plan::build_plan(in);
}

MemoryConfig create_regime_a_weight_memory_config(const ttnn::Shape& weight_shape, DataType dtype, IDevice* device) {
    // v1 supports only bf16 in1 (the reader + CBs are bf16). Reject other dtypes rather than accepting and
    // silently ignoring the argument — the shard byte layout below assumes a bf16 tile size.
    TT_FATAL(
        dtype == DataType::BFLOAT16,
        "create_regime_a_weight_memory_config supports only BFLOAT16 (only bf16 in1 is implemented), got {}",
        dtype);
    const uint32_t K = static_cast<uint32_t>(weight_shape[-2]);
    const uint32_t N = static_cast<uint32_t>(weight_shape[-1]);
    const uint32_t Kt = cdiv(K, TILE_HEIGHT);
    const uint32_t Nt = cdiv(N, TILE_WIDTH);

    // Config-independent + minimal padding: K is NOT padded (shard height = the tile-aligned K rows;
    // the balanced-tail reader never reads beyond valid K). N is padded up to a multiple of 8 tiles so
    // the width shard divides evenly across the 8 banks. Shard spec depends only on (K, N).
    const uint32_t Nt_pad = rup(Nt, kNumBanks);

    // Shard shape in ELEMENTS: Kt rows, ceil(Nt/8) columns per bank (width sharding across 8 banks).
    const std::array<uint32_t, 2> shard_shape = {Kt * TILE_HEIGHT, (Nt_pad / kNumBanks) * TILE_WIDTH};

    // Shard grid = the first 8 DRAM banks (Regime-A fixes G=8). NOTE: this assumes the target device
    // exposes >= 8 DRAM banks along the DRAM grid row (BH p150b = 8). Guard against smaller grids.
    const CoreCoord dram_grid = device->dram_grid_size();
    TT_FATAL(
        static_cast<uint32_t>(dram_grid.x) * static_cast<uint32_t>(dram_grid.y) >= kNumBanks,
        "regime_a_matmul in1 width-shard needs >= {} DRAM banks, device exposes {}x{}",
        kNumBanks,
        dram_grid.x,
        dram_grid.y);
    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{kNumBanks - 1, 0}));

    const ShardSpec shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR);
    return MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::DRAM, shard_spec);
}

}  // namespace ttnn::experimental::prim
