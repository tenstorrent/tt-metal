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

// Regime-A fixes the in1 DRAM shard to 8 banks (matches plan::Geometry::G).
constexpr uint32_t kNumBanks = 8u;

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t rup(uint32_t x, uint32_t y) { return cdiv(x, y) * y; }

// ---- Auto-selector (ported from tools/mm_sweep/picker_{table,v2}.py) ----
// v2 cost-model params (grid-search best on the 3262-config oracle: geomean 96.8%).
constexpr uint32_t kCsat = 24, kAcap = 6, kKbcap = 2;
constexpr double kKk = 0.5, kAa = 2.0, kOvl = 1.0, kStart = 0.0, kWst = 0.5;
constexpr uint32_t kL1Budget = 1440u * 1024u, kTB = 2048u;

// Lightweight geometry + feasibility (mirrors picker_v2.plan()). Returns false if infeasible.
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
    g.cores = 8u * Pk * Ns * Sm;
    if (g.cores < 16u || g.cores > 104u) {
        return false;
    }
    g.Ktl = rup(cdiv(Kt, Pk), kb * 8u);
    g.wasteK = static_cast<double>(Pk * g.Ktl) / Kt - 1.0;
    if (g.wasteK > 0.20) {
        return false;
    }
    g.Mblk = cdiv(Mt, Sm);
    const uint32_t Nband = cdiv(Nt, 8u);
    g.Nown = cdiv(Nband, Ns);
    if (nsb > g.Nown) {
        return false;
    }
    g.Nbpc = cdiv(g.Nown, nsb);
    g.wasteN = static_cast<double>(8u * Ns * g.Nbpc * nsb) / Nt - 1.0;
    if (g.wasteN > 0.20) {
        return false;
    }
    const uint32_t cb0 = g.Ktl * g.Mblk * kTB, cb1 = 4u * kb * nsb * kTB, cb2 = 2u * g.Mblk * nsb * kTB,
                   cb3 = g.Mblk * nsb * 4096u, cb7 = 2u * g.Mblk * nsb * kTB;
    return (cb0 + cb1 + cb2 + cb3 + cb7) <= kL1Budget;
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

}  // namespace

RegimeAMatmulConfig auto_select_config(uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    // Oracle lookup table (100% on the 20 FLUX/LTX production shapes), keyed by TILE dims (Mt,Kt,Nt).
    // value = {k_slices(Pk), n_slices(Ns), m_slices(Sm), k_block_tiles(kb), n_subblock_tiles(nsb)}.
    static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, RegimeAMatmulConfig> kTable = {
        {{1, 64, 16}, {4, 2, 1, 2, 1}},
        {{1, 64, 48}, {2, 2, 1, 4, 3}},
        {{1, 192, 48}, {6, 1, 1, 4, 2}},
        {{1, 64, 64}, {2, 2, 1, 4, 4}},
        {{1, 192, 72}, {4, 1, 1, 2, 9}},
        {{1, 192, 96}, {3, 1, 1, 4, 6}},
        {{1, 8, 192}, {1, 3, 1, 1, 8}},
        {{1, 192, 192}, {6, 1, 1, 4, 2}},
        {{1, 192, 288}, {3, 1, 1, 4, 6}},
        {{2, 192, 48}, {12, 1, 1, 2, 1}},
        {{2, 480, 48}, {12, 1, 1, 1, 3}},  // (64,15360,1536)
        {{2, 192, 144}, {6, 1, 1, 4, 2}},
        {{2, 144, 192}, {6, 1, 1, 1, 8}},
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
        {{8, 64, 32}, {4, 1, 2, 2, 2}},
    };
    if (auto it = kTable.find({Mt, Kt, Nt}); it != kTable.end()) {
        return it->second;
    }

    // Cost-model fallback: enumerate feasible (Sm=1) candidates, pick min cost.
    RegimeAMatmulConfig best{};
    double best_cost = std::numeric_limits<double>::infinity();
    const uint32_t Nband = cdiv(Nt, 8u);
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
                    if (c < best_cost) {
                        best_cost = c;
                        best = RegimeAMatmulConfig{
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
        best_cost != std::numeric_limits<double>::infinity(),
        "regime_a_matmul auto-select found no feasible config for Mt={} Kt={} Nt={}",
        Mt,
        Kt,
        Nt);
    return best;
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
    in.l1_budget_bytes = 1440u * 1024u;
    in.tb = 2048u;  // bf16 tile bytes
    in.tf = 4096u;  // fp32 tile bytes
    in.nn_chain = false;

    return plan::build_plan(in);
}

MemoryConfig create_regime_a_weight_memory_config(
    const ttnn::Shape& weight_shape, DataType /*dtype*/, IDevice* device) {
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
