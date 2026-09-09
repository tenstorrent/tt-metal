// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "small_m_matmul_config.hpp"

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
// small_m_matmul_plan.hpp — the SINGLE source of truth — and are reached here via the `plan::` alias.
using plan::kL1BudgetBytes;
using plan::kNumBanks;
using plan::kTileBytesBf16;
using plan::kTileBytesFp32;

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t rup(uint32_t x, uint32_t y) { return cdiv(x, y) * y; }

// ---- Auto-selector: measured lookup table + cost-model fallback ----
// Cost-model params (grid-search best on a 3262-config on-device sweep: geomean 96.8% of the measured optimum).
constexpr uint32_t kCsat = 24, kAcap = 6, kKbcap = 2;
constexpr double kKk = 0.5, kAa = 2.0, kOvl = 1.0, kStart = 0.0, kWst = 0.5;
// M-split fallback (fitted on Mt<=8 measurements): the Sm=1
// ranking is the ANCHOR; an Sm>1 candidate is chosen only for NARROW-N shapes (Nband<=kNbandMax, where
// N-split cannot supply parallelism) when its reduction-aware cost beats the anchor's by kMSplitMargin.
// kRk penalises split-K reduction (rk*(Pk-1)*out-tiles/core). Zero regression on the 60 measured shapes.
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
    const plan::FusionInputs& fu,
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
    // L1 via the shared authoritative sizer (plan::compute_cb_sizes), so the picker can never select a config
    // that build_plan() then rejects: it accounts for the chain's c_7 OR the reduce-scatter ring's c_8..c_10
    // (whichever the gate selects) plus any fused operand CBs c_4..c_6.
    return plan::compute_cb_sizes(Pk, Kt, g.Mblk, g.Ktl, nsb, kb, fu).l1_bytes <= kL1BudgetBytes;
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

// Min reduction-aware-cost config over Sm>1. Writes `best` and returns its cost, or infinity when no Sm>1
// config is feasible. Shared by the narrow-N hysteresis (choosing M-split over a feasible Sm=1 anchor) and by
// the large-Mt rescue (where NO Sm=1 config fits), so both rank candidates identically.
double best_msplit_config(
    uint32_t Mt, uint32_t Kt, uint32_t Nt, uint32_t Nband, const plan::FusionInputs& fu, SmallMMatmulConfig& best) {
    double best_cost = std::numeric_limits<double>::infinity();
    for (uint32_t Pk = 1; Pk <= 12u; ++Pk) {
        for (uint32_t Ns = 1; Ns <= 6u; ++Ns) {
            const uint32_t Nown = cdiv(Nband, Ns);
            for (uint32_t Sm = 2; Sm <= Mt; ++Sm) {
                for (uint32_t kb : {1u, 2u, 4u, 8u}) {
                    for (uint32_t nsb = 1; nsb <= Nown; ++nsb) {
                        PickGeo g{};
                        if (!pick_plan(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb, fu, g)) {
                            continue;
                        }
                        const double c = pick_cost_v3(Kt, Nt, Pk, kb, nsb, g);
                        if (c < best_cost) {
                            best_cost = c;
                            best = SmallMMatmulConfig{
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
    return best_cost;
}

}  // namespace

SmallMMatmulConfig auto_select_config(uint32_t Mt, uint32_t Kt, uint32_t Nt, const plan::FusionInputs& fu) {
    // Lookup table of measured winners (exhaustive on-device sweeps, Blackhole p150, unfused), keyed by TILE
    // dims (Mt,Kt,Nt). value = {k_slices(Pk), n_slices(Ns), m_slices(Sm), k_block_tiles(kb), n_subblock_tiles(nsb)}.
    // Trailing comment = logical M x K x N. Shapes not listed fall through to the cost model below.
    static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, SmallMMatmulConfig> kTable = {
        {{1, 64, 16}, {4, 2, 1, 2, 1}},     // 32x2048x512
        {{1, 64, 48}, {2, 2, 1, 4, 3}},     // 32x2048x1536
        {{1, 192, 48}, {6, 1, 1, 4, 2}},    // 32x6144x1536
        {{1, 64, 64}, {2, 2, 1, 4, 4}},     // 32x2048x2048
        {{1, 192, 72}, {3, 1, 1, 4, 5}},    // 32x6144x2304
        {{1, 192, 96}, {3, 1, 1, 4, 6}},    // 32x6144x3072
        {{1, 8, 192}, {1, 5, 1, 1, 5}},     // 32x256x6144
        {{1, 192, 192}, {6, 1, 1, 4, 2}},   // 32x6144x6144
        {{1, 192, 288}, {3, 1, 1, 4, 6}},   // 32x6144x9216
        {{2, 192, 48}, {3, 1, 1, 8, 2}},    // 64x6144x1536
        {{2, 480, 48}, {6, 1, 1, 2, 3}},    // 64x15360x1536
        {{2, 192, 144}, {6, 1, 1, 4, 2}},   // 64x6144x4608
        {{2, 144, 192}, {3, 2, 1, 2, 3}},   // 64x4608x6144
        {{2, 192, 288}, {6, 1, 1, 4, 2}},   // 64x6144x9216
        {{4, 192, 24}, {6, 1, 2, 4, 3}},    // 128x6144x768
        {{4, 480, 24}, {6, 1, 1, 2, 3}},    // 128x15360x768
        {{4, 192, 72}, {12, 1, 1, 2, 1}},   // 128x6144x2304
        {{4, 192, 144}, {12, 1, 1, 2, 1}},  // 128x6144x4608
        {{4, 72, 192}, {3, 2, 1, 1, 6}},    // 128x2304x6144
        {{16, 192, 48}, {6, 1, 2, 2, 3}},   // 512x6144x1536
        {{8, 64, 32}, {4, 1, 2, 2, 4}},     // 256x2048x1024
        {{8, 64, 192}, {4, 3, 1, 2, 2}},    // 256x2048x6144
        {{8, 480, 48}, {10, 1, 1, 2, 2}},   // 256x15360x1536
        {{8, 64, 16}, {4, 1, 2, 2, 2}},     // 256x2048x512
        {{1, 480, 24}, {6, 1, 1, 2, 3}},    // 32x15360x768
        {{8, 72, 192}, {3, 2, 2, 1, 6}},    // 256x2304x6144
        {{4, 64, 16}, {4, 1, 2, 2, 2}},     // 128x2048x512
        {{4, 480, 48}, {12, 1, 1, 1, 3}},   // 128x15360x1536
        {{2, 64, 16}, {4, 1, 1, 2, 2}},     // 64x2048x512
        {{8, 64, 48}, {4, 1, 2, 2, 6}},     // 256x2048x1536
        {{1, 64, 32}, {2, 4, 1, 4, 1}},     // 32x2048x1024
        {{2, 64, 32}, {4, 2, 1, 2, 2}},     // 64x2048x1024
        {{8, 480, 24}, {5, 1, 2, 4, 3}},    // 256x15360x768
        {{8, 64, 64}, {4, 1, 2, 2, 8}},     // 256x2048x2048
        {{4, 64, 32}, {4, 1, 2, 2, 4}},     // 128x2048x1024
        {{8, 192, 48}, {12, 1, 1, 2, 1}},   // 256x6144x1536
        {{1, 72, 192}, {3, 2, 1, 1, 6}},    // 32x2304x6144
        {{2, 64, 64}, {2, 3, 1, 2, 3}},     // 64x2048x2048
        {{1, 480, 48}, {6, 1, 1, 2, 3}},    // 32x15360x1536
        {{8, 192, 192}, {6, 1, 2, 2, 6}},   // 256x6144x6144
        {{4, 64, 64}, {4, 1, 2, 2, 8}},     // 128x2048x2048
        {{2, 480, 24}, {6, 1, 1, 2, 3}},    // 64x15360x768
        {{1, 192, 24}, {3, 1, 1, 4, 3}},    // 32x6144x768
        {{8, 192, 144}, {6, 1, 2, 2, 6}},   // 256x6144x4608
        {{16, 192, 24}, {6, 1, 2, 2, 3}},   // 512x6144x768
        {{16, 96, 192}, {6, 1, 2, 2, 6}},   // 512x3072x6144
        {{16, 72, 192}, {3, 2, 2, 1, 3}},   // 512x2304x6144
        {{16, 128, 160}, {4, 1, 3, 4, 5}},  // 512x4096x5120
        {{4, 64, 48}, {4, 2, 1, 2, 3}},     // 128x2048x1536
        {{2, 64, 48}, {4, 2, 1, 2, 3}},     // 64x2048x1536
        {{2, 192, 72}, {6, 1, 1, 4, 2}},    // 64x6144x2304
        {{2, 192, 24}, {3, 1, 1, 8, 3}},    // 64x6144x768
        {{1, 160, 40}, {5, 1, 1, 4, 3}},    // 32x5120x1280
    };
    if (auto it = kTable.find({Mt, Kt, Nt}); it != kTable.end()) {
        // Every table entry is a MEASURED unfused winner, so an unfused lookup is returned unconditionally --
        // byte-identical to before. Fused operands (c_4..c_6, c_10) add L1 that the measurements never paid
        // for, so when fusing we check the entry still fits and otherwise fall through to the cost model.
        // Falling through yields a slower-but-valid config instead of a planner TT_FATAL.
        const bool fusing = fu.has_bias || fu.has_ternary || fu.has_activation;
        if (!fusing) {
            return it->second;
        }
        const SmallMMatmulConfig& t = it->second;
        const uint32_t t_Ns = t.n_slices ? t.n_slices : 1u;
        const uint32_t t_nsb = t.n_subblock_tiles ? t.n_subblock_tiles : cdiv(cdiv(Nt, kNumBanks), t_Ns);
        PickGeo tg{};
        if (pick_plan(
                Mt,
                Kt,
                Nt,
                t_Ns,
                t.k_slices ? t.k_slices : 1u,
                t.m_slices ? t.m_slices : 1u,
                t.k_block_tiles ? t.k_block_tiles : 1u,
                t_nsb,
                fu,
                tg)) {
            return t;
        }
    }

    // Cost-model fallback. Step 1: the deployed Sm=1 ANCHOR (min deployed cost) -- unchanged behaviour.
    SmallMMatmulConfig anchor{};
    double anchor_cost = std::numeric_limits<double>::infinity();
    PickGeo anchor_g{};
    const uint32_t Nband = cdiv(Nt, kNumBanks);
    for (uint32_t Pk = 1; Pk <= 12u; ++Pk) {
        for (uint32_t Ns = 1; Ns <= 6u; ++Ns) {
            const uint32_t Nown = cdiv(Nband, Ns);
            for (uint32_t kb : {1u, 2u, 4u, 8u}) {
                for (uint32_t nsb = 1; nsb <= Nown; ++nsb) {
                    PickGeo g{};
                    if (!pick_plan(Mt, Kt, Nt, Ns, Pk, 1u, kb, nsb, fu, g)) {
                        continue;
                    }
                    const double c = pick_cost(Kt, Nt, kb, nsb, g);
                    if (c < anchor_cost) {
                        anchor_cost = c;
                        anchor_g = g;
                        anchor = SmallMMatmulConfig{
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
    // Step 1b: RESCUE where NO Sm=1 config is feasible. On large-Mt deep-K shapes the in0 k-slice-resident CB
    // is M_block * K_slice tiles, and at Sm=1 (M_block == Mt) that single buffer can exceed L1 for every
    // (Pk, Ns, kb, nsb) -- e.g. 512x15360x768 (Mt=16, Kt=480) needs >= 1.28 MB of a 1.41 MB budget for cb0
    // alone. M-split fixes exactly that by dividing M_block, so search Sm>1 rather than failing the op.
    // This is deliberately NOT behind the narrow-N gate below: that gate decides whether M-split is
    // PREFERABLE to a feasible Sm=1 pick, which is a different question from whether anything fits at all.
    // Behaviour is unchanged whenever an Sm=1 anchor exists, so no deployed pick moves.
    if (anchor_cost == std::numeric_limits<double>::infinity()) {
        // TWO STRUCTURALLY DIFFERENT REASONS land here, and they need different messages. The bank-interval
        // constraint is a function of Nt ALONE and pick_plan rejects on it before L1 is ever considered, so
        // reporting "no config fits L1" for a too-narrow N is simply wrong -- and narrow N is the case users
        // hit first. Check it explicitly so the error names the real cause and the real remedy.
        TT_FATAL(
            plan::nt_width_shard_feasible(Nt),
            // Report Nt, not a reconstructed N: only the tile count reaches here, and printing
            // Nt*TILE_WIDTH would quote a PADDED width back at a user who asked for something narrower
            // (N=8 would be reported as "N=32").
            "small_m_matmul cannot serve this N: in1 is DRAM width-sharded across {} banks, which requires "
            "7*ceil(Nt/8) < Nt so that every bank holds real data. This request rounds up to Nt={} tiles "
            "({} elements), where the trailing banks would be entirely padding. The smallest workable widths "
            "are Nt = 8, 15, 16, 22, 23, 24, 29.. (N = 256, 480, 512, 704, 736, 768, 928..). This is a "
            "SHAPE-DOMAIN limit of the small-M matmul, not a tuning failure: use a standard matmul for this N.",
            kNumBanks,
            Nt,
            Nt * tt::constants::TILE_WIDTH);
        SmallMMatmulConfig rescue{};
        const double rescue_cost = best_msplit_config(Mt, Kt, Nt, Nband, fu, rescue);
        TT_FATAL(
            rescue_cost != std::numeric_limits<double>::infinity(),
            "small_m_matmul cannot serve Mt={} Kt={} Nt={}: no Sm=1 config fits L1 ({} KB/core) and no "
            "M-split config is feasible either. This op keeps the in0 k-slice L1-RESIDENT, so cb0 alone is "
            "~(Mt/Sm)*(Kt/Pk) tiles, while core feasibility caps 8*Pk*Ns*Sm <= {} cores, i.e. Pk*Ns*Sm <= {}; "
            "shrinking cb0 needs a larger Sm*Pk than that cap allows, so at large Mt with deep K the two "
            "cannot both be satisfied (e.g. Mt=152, Kt=128 needs Sm*Pk >~ 32 against Pk*Ns*Sm <= {}). This is "
            "a SHAPE-DOMAIN limit of the small-M matmul, not a tuning failure: use a standard matmul for this shape.",
            Mt,
            Kt,
            Nt,
            plan::kL1BudgetBytes / 1024u,
            plan::kMaxCores,
            plan::kMaxCores / kNumBanks,
            plan::kMaxCores / kNumBanks);
        return rescue;
    }

    // Step 2: NARROW-N M-split hysteresis. Only where N-split cannot supply parallelism (Nband<=kNbandMax)
    // do we consider Sm>1, and only adopt it when its reduction-aware cost beats the anchor's by the margin.
    // Otherwise the anchor (deployed pick) is returned -> zero regression by construction.
    if (Nband > kNbandMax || Mt < 2u) {
        return anchor;
    }
    SmallMMatmulConfig bestG{};
    const double bestG_cost = best_msplit_config(Mt, Kt, Nt, Nband, fu, bestG);
    const double anchor_cost_v3 =
        pick_cost_v3(Kt, Nt, anchor.k_slices, anchor.k_block_tiles, anchor.n_subblock_tiles, anchor_g);
    if (bestG_cost < std::numeric_limits<double>::infinity() && bestG_cost < anchor_cost_v3 * (1.0 - kMSplitMargin)) {
        return bestG;
    }
    return anchor;
}

plan::PlanResult make_and_build_plan(
    IDevice* device,
    const Tensor& in0,
    const Tensor& in1,
    const std::optional<SmallMMatmulConfig>& cfg_opt,
    const plan::FusionInputs& fusion) {
    // Tile counts from logical shapes (tile = 32).
    const auto& a_shape = in0.logical_shape();
    const auto& w_shape = in1.logical_shape();
    const uint32_t Mt = cdiv(static_cast<uint32_t>(a_shape[-2]), TILE_HEIGHT);
    const uint32_t Kt = cdiv(static_cast<uint32_t>(a_shape[-1]), TILE_WIDTH);
    const uint32_t Nt = cdiv(static_cast<uint32_t>(w_shape[-1]), TILE_WIDTH);

    // config=None -> auto-select (deterministic in (Mt,Kt,Nt), so program-cache-safe: the cache key is
    // (nullopt config + tensor shapes) and the same shapes always resolve to the same config).
    const SmallMMatmulConfig cfg = cfg_opt.value_or(auto_select_config(Mt, Kt, Nt, fusion));

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
    // NOTE: we deliberately use the NOC_0 assignment for BOTH NoCs. The API returns "the worker to the RIGHT
    // of the DRAM controller" for either NoC, which is downstream (short) only for NOC_0: a NOC_1 read
    // response from a DRAM column travels -x and therefore wraps most of the way round the torus to reach a
    // worker on the column's right. So opt1 is not actually a good NOC_1 target, and measurement agrees -
    // using the true per-NoC assignment cost 0.6-3.7% on the shapes that still use these targets
    // (the mesh-placed shapes ignore them entirely).
    const auto opt1 = opt0;  // per the NOTE above, opt0 is the better target for BOTH NoCs

    plan::PlanInputs in;
    in.Mt = Mt;
    in.Kt = Kt;
    in.Nt = Nt;
    in.cfg = plan::SmallMConfig{
        .k_slices = cfg.k_slices,
        .n_slices = cfg.n_slices,
        .m_slices = cfg.m_slices,
        .k_block_tiles = cfg.k_block_tiles,
        .n_subblock_tiles = cfg.n_subblock_tiles};
    in.grid_x = static_cast<uint32_t>(grid.x);
    in.grid_y = static_cast<uint32_t>(grid.y);
    in.opt0 = opt0;
    in.opt1 = opt1;
    in.holes = {};       // v1: no explicit grid holes; find_near just walks to the next free logical core.
    in.fusion = fusion;  // fused operand CBs are real L1 -> the feasibility check must see them
    // Blackhole usable L1 ~1440 KB; the same budget the picker's feasibility check uses.
    in.l1_budget_bytes = kL1BudgetBytes;
    in.tb = kTileBytesBf16;  // bf16 tile bytes
    in.tf = kTileBytesFp32;  // fp32 tile bytes
    return plan::build_plan(in);
}

MemoryConfig create_small_m_weight_memory_config(const ttnn::Shape& weight_shape, DataType dtype, IDevice* device) {
    // v1 supports only bf16 in1 (the reader + CBs are bf16). Reject other dtypes rather than accepting and
    // silently ignoring the argument — the shard byte layout below assumes a bf16 tile size.
    TT_FATAL(
        dtype == DataType::BFLOAT16,
        "create_small_m_weight_memory_config supports only BFLOAT16 (only bf16 in1 is implemented), got {}",
        dtype);
    const uint32_t K = static_cast<uint32_t>(weight_shape[-2]);
    const uint32_t N = static_cast<uint32_t>(weight_shape[-1]);
    const uint32_t Kt = cdiv(K, TILE_HEIGHT);
    const uint32_t Nt = cdiv(N, TILE_WIDTH);

    // Reject a width the op itself cannot serve, HERE rather than at the first forward. Building a
    // valid-looking MemoryConfig for an infeasible N lets a caller lay a weight down at load time and
    // only discover at program build that no config exists -- by which point the layout choice is
    // already baked into the loaded parameter. Same bank-interval constraint pick_plan enforces.
    TT_FATAL(
        plan::nt_width_shard_feasible(Nt),
        "create_small_m_weight_memory_config cannot serve N={} (Nt={} tiles): in1 is DRAM width-sharded "
        "across {} banks, which requires 7*ceil(Nt/8) < Nt so that every bank holds real data. The "
        "smallest workable widths are Nt = 8, 15, 16, 22, 23, 24, 29.. (N = 256, 480, 512, 704, 736, "
        "768, 928..). This is a SHAPE-DOMAIN limit of the small-M matmul: use a standard matmul for "
        "this N. NOTE at tensor parallelism this is the LOCAL (per-device) N, not the global one.",
        N,
        Nt,
        kNumBanks);

    // Config-independent + minimal padding: K is NOT padded (shard height = the tile-aligned K rows;
    // the balanced-tail reader never reads beyond valid K). N is padded up to a multiple of 8 tiles so
    // the width shard divides evenly across the 8 banks. Shard spec depends only on (K, N).
    const uint32_t Nt_pad = rup(Nt, kNumBanks);

    // Shard shape in ELEMENTS: Kt rows, ceil(Nt/8) columns per bank (width sharding across 8 banks).
    const std::array<uint32_t, 2> shard_shape = {Kt * TILE_HEIGHT, (Nt_pad / kNumBanks) * TILE_WIDTH};

    // Shard grid = the first 8 DRAM banks (the op fixes G=8). NOTE: this assumes the target device
    // exposes >= 8 DRAM banks along the DRAM grid row (BH p150b = 8). Guard against smaller grids.
    const CoreCoord dram_grid = device->dram_grid_size();
    TT_FATAL(
        static_cast<uint32_t>(dram_grid.x) * static_cast<uint32_t>(dram_grid.y) >= kNumBanks,
        "small_m_matmul in1 width-shard needs >= {} DRAM banks, device exposes {}x{}",
        kNumBanks,
        dram_grid.x,
        dram_grid.y);
    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{kNumBanks - 1, 0}));

    const ShardSpec shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR);
    return MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::DRAM, shard_spec);
}

}  // namespace ttnn::experimental::prim
