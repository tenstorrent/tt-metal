// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <fmt/base.h>  // RINGCOST diagnostic line (physical ring-order experiment)

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/device.hpp>  // get_worker_noc_hop_distance (physical ring-order diag)

#include "regime_a_matmul_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in1_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/compute.cpp";

constexpr uint32_t kTileBytesBf16 = 2048u;
constexpr uint32_t kTileBytesFp32 = 4096u;

// Largest divisor of v that is <= cap (always >= 1).
uint32_t largest_div(uint32_t v, uint32_t cap) {
    if (v == 0) {
        return 1u;
    }
    for (uint32_t d = std::min(cap, v); d >= 1; --d) {
        if (v % d == 0) {
            return d;
        }
    }
    return 1u;
}

// mkcb: single-format circular buffer over a core range set (matches the harness form).
void mkcb(Program& program, const CoreRangeSet& crs, uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
    CircularBufferConfig c(ntiles * tsz, {{idx, df}});
    c.set_page_size(idx, tsz);
    CreateCircularBuffer(program, crs, c);
}

}  // namespace

RegimeAMatmulProgramFactory::cached_program_t RegimeAMatmulProgramFactory::create(
    const RegimeAMatmulParams& operation_attributes,
    const RegimeAMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Program program = CreateProgram();

    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    Tensor& out = tensor_return_value[0];  // chunk 0 (or the sole output when chunks==1)
    IDevice* device = in0.device();

    // Resolve config=None via the auto-selector (deterministic in the tile dims, program-cache-safe).
    const uint32_t Mt_r = (static_cast<uint32_t>(in0.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(in0.logical_shape()[-1]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = operation_attributes.config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));

    // ---- Run the pure host planner ----
    auto planres = make_and_build_plan(device, in0, in1, cfg);
    TT_FATAL(planres.ok(), "regime_a_matmul planner rejected config: {}", planres.error);
    plan::ExecutionPlan& P = *planres.plan;  // mutable: the ring-order diag overrides ring_pos/next/prev below
    const plan::Geometry& geo = P.geo;
    const plan::CbSizes& cb = P.cb;

    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    const uint32_t use_reduce = (Pk > 1u) ? 1u : 0u;

    // ---- Fused epilogue + output-split detection (all off => byte-identical no-fusion path). ----
    const bool has_bias = tensor_args.bias_tensor.has_value();
    const bool has_ternary = operation_attributes.fused_ternary_scalar.has_value();
    const bool has_activation = operation_attributes.fused_activation.has_value();
    const bool gate_is_fp32 = has_ternary && tensor_args.fused_ternary_input_b->dtype() == DataType::FLOAT32;
    // gate broadcast [1,N] vs full [M,N]. Decide from LOGICAL M, not padded: a full per-row gate with
    // M_logical in 2..32 pads to a single tile row, so padded_shape()/TILE_HEIGHT==1 cannot tell it apart
    // from a real [1,N] broadcast and would silently broadcast row 0 across all M rows. logical M==1 is the
    // only broadcast case, matching validate()'s tb_l[-2]==1 || tb_l[-2]==M check.
    const uint32_t broadcast_gate =
        has_ternary ? (tensor_args.fused_ternary_input_b->logical_shape()[-2] == 1u ? 1u : 0u) : 1u;
    const int32_t chunks = operation_attributes.chunks < 1 ? 1 : operation_attributes.chunks;
    const uint32_t n_chunks = static_cast<uint32_t>(chunks);
    const uint32_t out_ntc = Nt_r / n_chunks;  // per-chunk N tiles (validated divisible + tile-aligned)

    // Test-only diagnostic ablations: mask 0 (public path) => all three define maps are EMPTY, so the
    // compile is byte-identical to production. Each DIAG_* define is scoped to the kernel(s) that #ifdef it.
    const uint32_t diag = operation_attributes.diag_mask;
    std::map<std::string, std::string> rdefs;  // in1 reader
    std::map<std::string, std::string> wdefs;  // in0 ring/reduce writer
    std::map<std::string, std::string> ddefs;  // compute (added to cdefs below)
    if (diag & RegimeADiag::DIAG_SKIP_IN1_READ) {
        rdefs["DIAG_SKIP_IN1_READ"] = "1";
    }
    if (diag & RegimeADiag::DIAG_FWD_FLUSH_FIRST) {
        rdefs["DIAG_FWD_FLUSH_FIRST"] = "1";  // A/B baseline: OLD per-block flush-before-signal in1 forward
    }
    if (diag & RegimeADiag::DIAG_NO_COALESCE) {
        rdefs["DIAG_NO_COALESCE"] = "1";  // A/B baseline: OLD K_block per-row in1 reads (no coalescing)
    }
    if (diag & RegimeADiag::DIAG_SKIP_IN0_READ) {
        wdefs["DIAG_SKIP_IN0_READ"] = "1";
    }
    if (diag & RegimeADiag::DIAG_SKIP_IN0_FORWARD) {
        wdefs["DIAG_SKIP_IN0_FORWARD"] = "1";
    }
    if (diag & RegimeADiag::DIAG_NO_REDUCE) {
        wdefs["DIAG_NO_REDUCE"] = "1";
        ddefs["DIAG_NO_REDUCE"] = "1";
    }
    if (diag & RegimeADiag::DIAG_BARRIER_DRAIN) {
        wdefs["DIAG_BARRIER_DRAIN"] = "1";  // A/B baseline: OLD per-block phase-2 completion barrier
    }

    // ---- M-split worker PLACEMENT diagnostics (Sm>1; DEFAULT = the planner's CURRENT placement) ----
    // Overrides only P.cores[i].coord (logical indices / ownership / the factory reader->i+s & slave->i-mm arg
    // math are unchanged). Runs BEFORE the ring reorder below so PARETO recomputes on the new coords.
    // READERS_FIRST: place every mm==0 DRAM reader (same bank targets / NoC / logical-Manhattan spiral) before
    // any slave. IN1_NEAR (implies readers-first): place each slave at the free worker minimizing the directed
    // reader->slave hop distance on the group's in1-reader NoC. Emits PLACECOST (gated by TT_MM_PLACECOST).
    const bool place_current = (diag & RegimeADiag::DIAG_PLACE_CURRENT) != 0u;              // diag: planner's placement
    const bool place_readers_first = (diag & RegimeADiag::DIAG_PLACE_READERS_FIRST) != 0u;  // diag: bank-spiral slaves
    const bool place_in1_near = !place_readers_first;  // DEFAULT = in1-near slaves; readers-first bit -> bank spiral
    if (!place_current && Sm > 1u) {  // DEFAULT (and readers-first diag) re-place; DIAG_PLACE_CURRENT keeps planner's
        namespace expd = tt::tt_metal::experimental::Device;
        const uint32_t preaders = geo.num_cores / 8u;
        const CoreCoord grid = device->compute_with_storage_grid_size();
        const auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
        const auto opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1);
        std::set<std::pair<uint32_t, uint32_t>> used;
        auto bank_tgt = [&](uint32_t b, uint32_t noc) { return noc ? opt1[b] : opt0[b]; };
        // logical-Manhattan spiral over the compute grid (mirrors the planner's find_near).
        auto find_near = [&](CoreCoord t) -> CoreCoord {
            for (int d = 0; d < (int)(grid.x + grid.y); ++d) {
                for (int dx = -d; dx <= d; ++dx) {
                    const int rem = d - (dx < 0 ? -dx : dx);
                    for (int sgn = 0; sgn <= 1; ++sgn) {
                        const int dy = sgn ? -rem : rem;
                        const int x = (int)t.x + dx, y = (int)t.y + dy;
                        if (x < 0 || y < 0 || (uint32_t)x >= grid.x || (uint32_t)y >= grid.y) {
                            continue;
                        }
                        const auto key = std::make_pair((uint32_t)x, (uint32_t)y);
                        if (used.count(key)) {
                            continue;
                        }
                        used.insert(key);
                        return CoreCoord{(uint32_t)x, (uint32_t)y};
                    }
                }
            }
            return CoreCoord{t.x, t.y};
        };
        auto set_coord = [&](uint32_t i, CoreCoord c) {
            P.cores[i].coord.x = c.x;
            P.cores[i].coord.y = c.y;
        };
        // pass 1: every mm==0 reader around its bank target (readers-first).
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t p = 0; p < preaders; ++p) {
                const uint32_t i = b * preaders + p;
                if (P.cores[i].mm == 0u) {
                    set_coord(i, find_near(bank_tgt(b, P.cores[i].noc)));
                }
            }
        }
        // pass 2: slaves — IN1_NEAR minimizes directed reader->slave hop on the reader NoC; else bank spiral.
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t p = 0; p < preaders; ++p) {
                const uint32_t i = b * preaders + p;
                if (P.cores[i].mm == 0u) {
                    continue;
                }
                if (place_in1_near) {
                    const uint32_t ri = i - P.cores[i].mm;  // this group's reader (contiguous index)
                    const CoreCoord rc{P.cores[ri].coord.x, P.cores[ri].coord.y};
                    const NOC rnoc = P.cores[i].noc ? NOC::NOC_1 : NOC::NOC_0;
                    CoreCoord best{};
                    uint32_t bestd = 0xffffffffu;
                    bool found = false;
                    for (uint32_t y = 0; y < grid.y; ++y) {
                        for (uint32_t x = 0; x < grid.x; ++x) {
                            if (used.count(std::make_pair(x, y))) {
                                continue;
                            }
                            const uint32_t dd = expd::get_worker_noc_hop_distance(device, rc, CoreCoord{x, y}, rnoc);
                            if (!found || dd < bestd) {
                                bestd = dd;
                                best = CoreCoord{x, y};
                                found = true;
                            }
                        }
                    }
                    used.insert(std::make_pair(best.x, best.y));
                    set_coord(i, best);
                } else {
                    set_coord(i, find_near(bank_tgt(b, P.cores[i].noc)));
                }
            }
        }
        // PLACECOST: per group, reader's dist from its bank target + directed reader->slave hops (in1 NoC).
        if (std::getenv("TT_MM_PLACECOST") != nullptr) {
            for (uint32_t b = 0; b < 8u; ++b) {
                for (uint32_t p = 0; p < preaders; ++p) {
                    const uint32_t i = b * preaders + p;
                    if (P.cores[i].mm != 0u) {
                        continue;
                    }
                    const CoreCoord rc{P.cores[i].coord.x, P.cores[i].coord.y};
                    const NOC rnoc = P.cores[i].noc ? NOC::NOC_1 : NOC::NOC_0;
                    const CoreCoord tgt = bank_tgt(b, P.cores[i].noc);
                    std::string fwd;
                    uint32_t maxf = 0;
                    for (uint32_t s = 1; s < Sm; ++s) {
                        const CoreCoord sc{P.cores[i + s].coord.x, P.cores[i + s].coord.y};
                        const uint32_t dd = expd::get_worker_noc_hop_distance(device, rc, sc, rnoc);
                        fwd += std::to_string(dd) + (s + 1 < Sm ? "," : "");
                        maxf = std::max(maxf, dd);
                    }
                    fmt::print(
                        "PLACECOST b={} p={} noc={} reader=({},{}) tgt=({},{}) rdr2tgt={} fwd=[{}] maxfwd={}\n",
                        b,
                        p,
                        P.cores[i].noc,
                        rc.x,
                        rc.y,
                        tgt.x,
                        tgt.y,
                        expd::get_worker_noc_hop_distance(device, tgt, rc, rnoc),
                        fwd,
                        maxf);
                }
            }
        }
    }

    // ---- Physical-topology-aware in0 ring ordering (test-only; DEFAULT = bank order [0..7]) ----
    // Overrides ring_pos/ring_next_idx/ring_prev_idx per ring group using the group's WRITER NoC authoritative
    // hop distance (get_worker_noc_hop_distance, logical->physical + directed torus routing w/ wraparound).
    // Placement/work/reduction are unchanged; only the ring visiting order (which core seeds which in0 shard,
    // the forward route, and the in1 rotated read) changes — correct for ANY permutation. Emits a RINGCOST
    // line per group (bank/greedy/opt max+total edge cost) for the report. No effect on the public path.
    const bool ring_bank = (diag & RegimeADiag::DIAG_RING_BANK) != 0u;        // diagnostic: bank order [0..7]
    const bool ring_opt_mm0 = (diag & RegimeADiag::DIAG_RING_OPT_MM0) != 0u;  // diagnostic: mm==0-only objective
    const bool ring_total = (diag & RegimeADiag::DIAG_RING_TOTAL) != 0u;      // diagnostic: total_maxedge
    const bool ring_maxedge = (diag & RegimeADiag::DIAG_RING_MAXEDGE) != 0u;  // diagnostic: maxedge_total
    if (!ring_bank) {  // DEFAULT = PARETO across all Sm mm-rings; other objectives if selected
        namespace expdev = tt::tt_metal::experimental::Device;
        const uint32_t preaders = geo.num_cores / 8u;
        // directed route cost of one 8-core cycle over a single ring's hop matrix: (max edge, total hops).
        auto ring_cost = [](const std::array<uint32_t, 8>& ord,
                            const std::array<std::array<uint32_t, 8>, 8>& d) -> std::pair<uint32_t, uint32_t> {
            uint32_t mx = 0, tot = 0;
            for (uint32_t p = 0; p < 8u; ++p) {
                const uint32_t e = d[ord[p]][ord[(p + 1u) % 8u]];
                tot += e;
                mx = std::max(mx, e);
            }
            return {mx, tot};
        };
        // M-split (Sm>1): slices differing only in mm form a (kk,nn) group of Sm CONTIGUOUS slice indices
        // [base, base+Sm), all sharing the same writer NoC. Their in1 slaves receive in1 in the mm==0 READER's
        // shard order while their in0 rings are separate physical cores, so the WHOLE group MUST use the SAME
        // permutation (reader/slave ring_pos must agree per bank) or the in0/in1 pairing corrupts.
        // OBJECTIVE (default): lexicographic over the Sm physical mm-rings — (1) minimize the worst directed
        // edge across ALL rings, (2) then the summed hops across ALL edges AND rings. This accounts for the
        // slaves' routes, not just the reader's. DIAG_RING_OPT_MM0 reverts to scoring only the mm==0 ring.
        for (uint32_t base = 0; base < preaders; base += Sm) {
            // shared writer NoC (opposite the reader's): noc==0 -> writer NOC1; noc==1 -> writer NOC0.
            const NOC wnoc = (P.cores[base].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
            // one 8x8 hop matrix per mm-ring (same wnoc, different physical cores).
            std::vector<std::array<std::array<uint32_t, 8>, 8>> dm(Sm);
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                auto lc = [&](uint32_t b) {
                    const auto& c = P.cores[b * preaders + base + mm].coord;
                    return CoreCoord{c.x, c.y};
                };
                for (uint32_t a = 0; a < 8u; ++a) {
                    for (uint32_t b = 0; b < 8u; ++b) {
                        dm[mm][a][b] = (a == b) ? 0u : expdev::get_worker_noc_hop_distance(device, lc(a), lc(b), wnoc);
                    }
                }
            }
            // per-candidate metrics across all Sm mm-rings: ring0 (mm==0) max/total; aggmax = worst edge over
            // rings; aggtot = summed hops over all rings; maxringtot = worst per-ring total.
            struct Metrics {
                uint32_t r0max, r0tot, aggmax, aggtot, maxringtot;
            };
            auto metrics = [&](const std::array<uint32_t, 8>& ord) -> Metrics {
                Metrics m{0, 0, 0, 0, 0};
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    const auto [rm, rt] = ring_cost(ord, dm[mm]);
                    if (mm == 0) {
                        m.r0max = rm;
                        m.r0tot = rt;
                    }
                    m.aggmax = std::max(m.aggmax, rm);
                    m.aggtot += rt;
                    m.maxringtot = std::max(m.maxringtot, rt);
                }
                return m;
            };
            const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
            // exhaustive: fix bank 0 at pos 0, permute the other 7 (5040 cycles; directed => both orientations).
            // Pass 1 tracks the lexicographic objectives (each first-strict-min wins). We also record the
            // aggtot of the MM0-selected order as the PARETO budget.
            std::array<uint32_t, 8> opt_mm0 = bank, opt_maxedge = bank, opt_total = bank, opt_pareto = bank;
            auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
                return a0 < b0 || (a0 == b0 && a1 < b1);
            };
            {
                std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
                Metrics b_mm0{~0u, ~0u, ~0u, ~0u, ~0u}, b_me{~0u, ~0u, ~0u, ~0u, ~0u}, b_to{~0u, ~0u, ~0u, ~0u, ~0u};
                auto cand_of = [](const std::array<uint32_t, 7>& t) {
                    std::array<uint32_t, 8> c{};
                    c[0] = 0;
                    for (uint32_t i = 0; i < 7u; ++i) {
                        c[i + 1u] = t[i];
                    }
                    return c;
                };
                do {
                    const std::array<uint32_t, 8> cand = cand_of(tail);
                    const Metrics m = metrics(cand);
                    if (lt2(m.r0max, m.r0tot, b_mm0.r0max, b_mm0.r0tot)) {
                        b_mm0 = m;
                        opt_mm0 = cand;
                    }
                    if (lt2(m.aggmax, m.aggtot, b_me.aggmax, b_me.aggtot)) {
                        b_me = m;
                        opt_maxedge = cand;
                    }
                    if (lt2(m.aggtot, m.aggmax, b_to.aggtot, b_to.aggmax)) {
                        b_to = m;
                        opt_total = cand;
                    }
                } while (std::next_permutation(tail.begin(), tail.end()));

                // Pass 2 — PARETO: min aggmax (then aggtot) subject to aggtot <= MM0's aggtot (never worse
                // total than MM0). Seeded with MM0 itself (satisfies the constraint by construction).
                opt_pareto = opt_mm0;
                Metrics b_pa = b_mm0;
                const uint32_t budget = b_mm0.aggtot;
                std::array<uint32_t, 7> tail2 = {1, 2, 3, 4, 5, 6, 7};
                do {
                    const std::array<uint32_t, 8> cand = cand_of(tail2);
                    const Metrics m = metrics(cand);
                    if (m.aggtot <= budget && lt2(m.aggmax, m.aggtot, b_pa.aggmax, b_pa.aggtot)) {
                        b_pa = m;
                        opt_pareto = cand;
                    }
                } while (std::next_permutation(tail2.begin(), tail2.end()));
            }
            // DEFAULT = PARETO (chosen after the two-run objective A/B); diagnostics select the others.
            const std::array<uint32_t, 8>& sel = ring_opt_mm0   ? opt_mm0
                                                 : ring_total   ? opt_total
                                                 : ring_maxedge ? opt_maxedge
                                                                : opt_pareto;
            // apply the selected order to ALL Sm slices of this group (same permutation => reader/slave ring_pos
            // agree per bank, preserving in0/in1 pairing under M-split).
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                const uint32_t jj = base + mm;
                for (uint32_t pos = 0; pos < 8u; ++pos) {
                    const uint32_t ci = sel[pos] * preaders + jj;
                    P.cores[ci].ring_pos = pos;
                    P.cores[ci].ring_next_idx = sel[(pos + 1u) % 8u] * preaders + jj;
                    P.cores[ci].ring_prev_idx = sel[(pos + 7u) % 8u] * preaders + jj;
                }
            }
            // RINGCOST diagnostic (route costs, gated behind TT_MM_RINGCOST so the production path is silent on
            // compile). Reports each candidate's GROUP-AGGREGATE cost (worst edge over the Sm rings; summed hops
            // over all rings) PLUS the selected order's PER-RING (max,total) breakdown so the report can
            // distinguish per-ring from aggregated-across-rings costs. The harness further aggregates across
            // the (kk,nn) groups of the whole op.
            if (std::getenv("TT_MM_RINGCOST") != nullptr) {
                auto join = [](const std::array<uint32_t, 8>& o) {
                    std::string s;
                    for (uint32_t p = 0; p < 8u; ++p) {
                        s += std::to_string(o[p]) + (p + 1u < 8u ? "," : "");
                    }
                    return s;
                };
                // group-aggregate (aggmax:aggtot) of each candidate; also the selected order's per-ring maxes.
                auto ac = [&](const std::array<uint32_t, 8>& o) {
                    const Metrics m = metrics(o);
                    return std::to_string(m.aggmax) + ":" + std::to_string(m.aggtot) + ":" +
                           std::to_string(m.maxringtot);
                };
                const char* selname = ring_opt_mm0 ? "mm0" : ring_total ? "total" : ring_maxedge ? "maxedge" : "pareto";
                std::string perring;
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    const auto [m, t] = ring_cost(sel, dm[mm]);
                    perring += "(" + std::to_string(m) + ":" + std::to_string(t) + ")";
                }
                // fields are aggmax:aggtot:maxringtot per candidate (group-aggregate). op-level aggregation is
                // done by the harness across (kk,nn) groups.
                fmt::print(
                    "RINGCOST group={} Sm={} wnoc={} sel={} bank[{}]={} mm0[{}]={} maxedge[{}]={} total[{}]={} "
                    "pareto[{}]={} sel_perring={}\n",
                    base,
                    Sm,
                    (wnoc == NOC::NOC_0 ? 0 : 1),
                    selname,
                    join(bank),
                    ac(bank),
                    join(opt_mm0),
                    ac(opt_mm0),
                    join(opt_maxedge),
                    ac(opt_maxedge),
                    join(opt_total),
                    ac(opt_total),
                    join(opt_pareto),
                    ac(opt_pareto),
                    perring);
            }
        }
    }
    if (diag & RegimeADiag::DIAG_LOCAL_FEED) {
        rdefs["DIAG_LOCAL_FEED"] = "1";
        wdefs["DIAG_LOCAL_FEED"] = "1";
        ddefs["DIAG_LOCAL_FEED"] = "1";
    }
    if (diag & RegimeADiag::DIAG_FULL_IN0_WAIT) {
        ddefs["DIAG_FULL_IN0_WAIT"] = "1";  // A/B baseline: old full-slice startup barrier (compute-only)
    }
    const bool scatter = (diag & RegimeADiag::DIAG_IN0_SCATTER) != 0u;
    if (scatter) {
        wdefs["DIAG_IN0_SCATTER"] = "1";  // writer phase-1 uses direct scatter; needs the G-1 ahead peers (below)
    }
    // Replicated shorter ring: IN0_REPL (2 or 4) goes to BOTH the writer (seed reads + R-bundle rotation) and
    // the in1 reader (matching shard order). No runtime-arg change (nearest-neighbor forward reuses fwd_next).
    if (diag & RegimeADiag::DIAG_IN0_REPL2) {
        wdefs["IN0_REPL"] = "2";
        rdefs["IN0_REPL"] = "2";
    } else if (diag & RegimeADiag::DIAG_IN0_REPL4) {
        wdefs["IN0_REPL"] = "4";
        rdefs["IN0_REPL"] = "4";
    }

    // ---- Fused-epilogue / output-split kernel defines (empty => byte-identical no-fusion compile). ----
    // Compute-only fusion defines are collected here and merged into cdefs at compute-kernel creation.
    std::map<std::string, std::string> fdefs_compute;
    if (has_bias) {
        wdefs["FUSE_BIAS"] = "1";
        fdefs_compute["FUSE_BIAS"] = "1";
    }
    if (has_ternary) {
        wdefs["FUSE_TERNARY"] = "1";
        fdefs_compute["FUSE_TERNARY"] = "1";
        if (gate_is_fp32) {
            wdefs["TERNARY_B_IS_FLOAT32"] = "1";
            fdefs_compute["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }
    if (n_chunks > 1u) {
        wdefs["OUT_CHUNKS"] = "1";
    }
    if (has_activation) {
        auto act = ttnn::operations::unary::utils::get_defines(
            operation_attributes.fused_activation->op_type,
            operation_attributes.fused_activation->params,
            "ACTIVATION",
            "fused_act_dst_id",
            out.dtype());
        fdefs_compute.insert(act.begin(), act.end());
    }

    // ---- Core range sets: all cores + split-NoC groups (g0 = noc 0, g1 = noc 1) ----
    std::set<CoreRange> all_set, g0_set, g1_set;
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> core_noc;
    cores.reserve(geo.num_cores);
    core_noc.reserve(geo.num_cores);
    for (const auto& cp : P.cores) {
        CoreCoord c{cp.coord.x, cp.coord.y};
        cores.push_back(c);
        core_noc.push_back(cp.noc);
        all_set.insert(CoreRange(c, c));
        (cp.noc ? g1_set : g0_set).insert(CoreRange(c, c));
    }
    CoreRangeSet all_cores(all_set);
    CoreRangeSet g0(g0_set);
    CoreRangeSet g1(g1_set);

    // ---- Circular buffers (spec §5) on all cores ----
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in0 k-slice resident
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in1 (depth 4)
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // out
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);    // fp32 intermediate
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // reduce (Pk>1 only)
    }
    // Fused-epilogue operand CBs (only when the matching fusion is active). c_4 bias [1,N_sub], c_5 residual
    // [M,N] block, c_6 gate [1,N_sub] (broadcast) or [M,N] block. Sized to hold a full sub-block so the
    // writer can stream all M rows while compute consumes them (matches minimal_matmul's ternary CB sizing).
    const uint32_t out_blk_tiles = geo.M_block_capacity * geo.N_sub;
    if (has_bias) {
        mkcb(program, all_cores, 4, geo.N_sub, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
    if (has_ternary) {
        mkcb(program, all_cores, 5, out_blk_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        const tt::DataFormat gfmt = gate_is_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t gtsz = gate_is_fp32 ? kTileBytesFp32 : kTileBytesBf16;
        const uint32_t gate_tiles = broadcast_gate ? geo.N_sub : out_blk_tiles;
        mkcb(program, all_cores, 6, gate_tiles, gfmt, gtsz);
    }

    // ---- Semaphores ----
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);      // in0 ring recv
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);      // reduction recv
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);  // cb_reduce reverse credit
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;                         // M-split reader<->slaves
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
    }
    // DIAG_IN0_XCHG: G-1 per-slot readiness semaphores so the writer pushes each in0 slot the moment its
    // direct-exchange write lands (incremental overlap), rather than one ring counter. Created only for the
    // xchg program so the public path's semaphore layout is unchanged.
    const bool xchg = (diag & RegimeADiag::DIAG_IN0_XCHG) != 0u;
    const bool xchgrr = (diag & RegimeADiag::DIAG_IN0_XCHGRR) != 0u;
    std::vector<uint32_t> xchg_slotsem;
    if (xchg || xchgrr) {  // both direct-exchange schedules use G-1 per-slot readiness semaphores
        for (uint32_t d = 1; d < geo.G; ++d) {
            xchg_slotsem.push_back(CreateSemaphore(program, all_cores, 0u));
        }
        wdefs[xchg ? "DIAG_IN0_XCHG" : "DIAG_IN0_XCHGRR"] = "1";
    }

    // ---- Kernels ----
    // in1 reader == consumer. compile args (in1_reader.cpp order). No TensorAccessorArgs.
    std::vector<uint32_t> rct = {
        kb,                      // 0 K_block
        geo.N_sub,               // 1 N_block
        geo.W,                   // 2 W
        geo.G,                   // 3 G (=8)
        kTileBytesBf16,          // 4 tile_bytes
        geo.N_bpc,               // 5 N_bpc
        geo.in1_shard_stride_n,  // 6 in1_shard_stride_n (physical per-bank width)
        in1valid_sem,            // 7
        in1ready_sem,            // 8
        0u};                     // 9 in1_mcast (0 for v1: unicast forward)

    auto mk = [&](const char* src,
                  const CoreRangeSet& g,
                  DataMovementProcessor proc,
                  NOC noc,
                  const std::vector<uint32_t>& ct,
                  const std::map<std::string, std::string>& defs) -> KernelHandle {
        if (g.num_cores() == 0) {
            return 0;
        }
        return CreateKernel(
            program, src, g, DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct, .defines = defs});
    };

    // writer compile args (in0_ring_reduce_writer.cpp order). TensorAccessorArgs(in0) then (out).
    std::vector<uint32_t> wct = {
        geo.M_block_capacity,  // 0
        kb,                    // 1 K_block
        geo.N_sub,             // 2 N_block
        geo.K_num_blocks_eff,  // 3 K_num_blocks
        kTileBytesBf16,        // 4 tile_bytes
        geo.in0_stride_k,      // 5 in0 row stride (physical = Kt)
        geo.out_stride_n,      // 6 out row stride (physical = Nt)
        geo.W,                 // 7 W
        geo.G,                 // 8 G
        fwd_sem,               // 9
        red_sem,               // 10
        geo.N_bpc,             // 11 N_bpc
        redfree_sem,           // 12
        use_reduce};           // 13
    TensorAccessorArgs(*in0.buffer()).append_to(wct);
    TensorAccessorArgs(*out.buffer()).append_to(wct);
    // Fused-operand accessors, in the order the writer kernel expects: bias, then residual/gate.
    if (has_bias) {
        TensorAccessorArgs(*tensor_args.bias_tensor->buffer()).append_to(wct);
    }
    if (has_ternary) {
        TensorAccessorArgs(*tensor_args.fused_ternary_input_a->buffer()).append_to(wct);
        TensorAccessorArgs(*tensor_args.fused_ternary_input_b->buffer()).append_to(wct);
    }

    // Split-NOC: reader on the core's in1 NoC, writer on the OTHER NoC.
    //   g0 (noc==0): reader RISCV_0/NOC0, writer RISCV_1/NOC1
    //   g1 (noc==1): reader RISCV_1/NOC1, writer RISCV_0/NOC0
    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, rdefs);
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, rdefs);
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // compute (spec §6c). fp32 DST limit: subblock_h * subblock_w <= 4.
    const uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    const uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
    std::vector<uint32_t> cct = {
        geo.K_num_blocks_eff,  // 0 K_num_blocks
        geo.M_block_capacity,  // 1 M_block_tiles
        kb,                    // 2 K_block_tiles
        geo.N_sub,             // 3 N_block_tiles
        1u,                    // 4 M_blocks_per_core
        geo.N_bpc,             // 5 N_blocks_per_core
        sbh,                   // 6 subblock_h
        sbw};                  // 7 subblock_w
    std::map<std::string, std::string> cdefs = {{"REDUCE_K", "1"}, {"IN0_KSLICE_RESIDENT", "1"}};
    cdefs.insert(ddefs.begin(), ddefs.end());                  // test-only diagnostic defines (empty for mask 0)
    cdefs.insert(fdefs_compute.begin(), fdefs_compute.end());  // fusion defines (empty for the no-fusion path)
    KernelHandle compute = CreateKernel(
        program,
        kComputeKernel,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = cct,
            .defines = cdefs});

    // ---- Runtime args ----
    const uint32_t in0_addr = in0.buffer()->address();
    const uint32_t in1_addr = in1.buffer()->address();
    const uint32_t out_addr = out.buffer()->address();

    auto phys = [&](uint32_t core_idx) {
        const auto& c = P.cores[core_idx].coord;
        return device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
    };

    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const plan::CorePlan& cp = P.cores[i];
        const KernelHandle rh = cp.noc ? readerB : readerA;
        const KernelHandle wh = cp.noc ? writerB : writerA;

        // in1 reader runtime args.
        std::vector<uint32_t> ra = {
            in1_addr,     // 0
            cp.bank,      // 1
            cp.ring_pos,  // 2
            cp.k_start,   // 3 first logical K tile (balanced)
            cp.n_local,   // 4 within-bank column offset
            cp.valid_k,   // 5 valid K tiles (rest of capacity zero-filled)
            cp.valid_n};  // 6 valid N tiles this core owns
        if (Sm == 1u) {
            ra.push_back(2u);  // 5 mrole = solo
            ra.push_back(0u);  // 6 mpeers
        } else if (cp.mm == 0u) {
            // reader (mm==0 of this (bank,kk,nn) group): read from DRAM + forward to the Sm-1 slaves.
            ra.push_back(1u);       // mrole = reader
            ra.push_back(Sm - 1u);  // mpeers
            for (uint32_t s = 1; s < Sm; ++s) {
                auto p = phys(i + s);  // slaves are the next Sm-1 contiguous core indices (mm innermost)
                ra.push_back(p.x);
                ra.push_back(p.y);
            }
        } else {
            // slave: receive from the group's reader (core i - mm).
            ra.push_back(0u);  // mrole = slave
            ra.push_back(1u);  // mpeers
            auto p = phys(i - cp.mm);
            ra.push_back(p.x);
            ra.push_back(p.y);
        }
        SetRuntimeArgs(program, rh, cores[i], ra);

        // writer runtime args.
        auto fwd_next = phys(cp.ring_next_idx);
        auto red_next = phys(cp.red_next_idx);
        auto red_prev = phys(cp.red_prev_idx);
        std::vector<uint32_t> wa = {
            in0_addr,                // 0
            out_addr,                // 1
            cp.m_start,              // 2 first logical M tile (balanced)
            cp.n_start,              // 3 first logical (global) N tile (output addressing)
            cp.k_start,              // 4 first logical K tile (balanced)
            cp.ring_pos,             // 5
            fwd_next.x,              // 6
            fwd_next.y,              // 7
            red_next.x,              // 8
            red_next.y,              // 9
            cp.is_bottom ? 1u : 0u,  // 10
            cp.is_top ? 1u : 0u,     // 11
            red_prev.x,              // 12
            red_prev.y,              // 13
            cp.valid_k,              // 14 valid K tiles (rest of capacity zero)
            cp.valid_m,              // 15 valid M tiles (rest zero / not written)
            cp.valid_n};             // 16 valid N tiles (rest zero / not written)
        // DIAG_IN0_SCATTER (test-only variant): append the G-1 ring peers AHEAD of this core (args 17..),
        // in d-ahead order (ring_next^1..ring_next^{G-1}). Core scatters its own shard to peer d's cb0 slot
        // d; each peer receives from the core d-behind, reproducing the ring's cb0 layout. Appended only for
        // the scatter program (distinct hash), so the public path's arg layout is unchanged.
        if (scatter || xchg || xchgrr) {
            // XCHG/XCHGRR: prepend the G-1 per-slot semaphore IDs (args 17..17+G-2). Then (scatter/xchg/
            // xchgrr) the G-1 ahead peers in d-ahead order (ring_next^1..). Core writes its own shard to peer
            // d's slot d and signals that peer's slot-d sem. Appended only for these programs; public-path
            // arg layout unchanged.
            if (xchg || xchgrr) {
                for (uint32_t d = 1; d < geo.G; ++d) {
                    wa.push_back(xchg_slotsem[d - 1]);
                }
            }
            uint32_t cur = i;
            for (uint32_t d = 1; d < geo.G; ++d) {
                cur = P.cores[cur].ring_next_idx;
                auto pc = phys(cur);
                wa.push_back(pc.x);
                wa.push_back(pc.y);
            }
        }
        // Fused-epilogue / output-split writer args (index 17+). Never combined with the diag ablations above
        // (those only run via the internal diag entry, which passes no fusion). Order MUST match the writer
        // kernel's fidx reads: bias, then residual/gate/broadcast, then chunk count/width/addresses.
        if (has_bias) {
            wa.push_back(tensor_args.bias_tensor->buffer()->address());
        }
        if (has_ternary) {
            wa.push_back(tensor_args.fused_ternary_input_a->buffer()->address());
            wa.push_back(tensor_args.fused_ternary_input_b->buffer()->address());
            wa.push_back(broadcast_gate);
        }
        if (n_chunks > 1u) {
            wa.push_back(n_chunks);
            wa.push_back(out_ntc);
            for (uint32_t c = 1; c < n_chunks; ++c) {
                wa.push_back(tensor_return_value[c].buffer()->address());
            }
        }
        SetRuntimeArgs(program, wh, cores[i], wa);

        // compute runtime args: fixed rectangular block over the schedule capacities. N_end spans ALL
        // N_bpc sub-blocks (spec §7); zero-filled tail positions contribute zero. When a fusion is active the
        // reduction-root flag (is_top) follows, then the addcmul scalar bits + gate-broadcast flag.
        std::vector<uint32_t> ca = {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u};
        if (has_bias || has_ternary || has_activation) {
            ca.push_back(cp.is_top ? 1u : 0u);
        }
        if (has_ternary) {
            const float sc = *operation_attributes.fused_ternary_scalar;
            ca.push_back(*reinterpret_cast<const uint32_t*>(&sc));
            ca.push_back(broadcast_gate);
        }
        SetRuntimeArgs(program, compute, cores[i], ca);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .num_cores = geo.num_cores,
            .cores = std::move(cores),
            .core_noc = std::move(core_noc),
            .readerA = readerA,
            .readerB = readerB,
            .writerA = writerA,
            .writerB = writerB,
            .compute = compute,
            .has_bias = has_bias,
            .has_ternary = has_ternary,
            .n_chunks = n_chunks}};
}

void RegimeAMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RegimeAMatmulParams& /*operation_attributes*/,
    const RegimeAMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = tensor_return_value[0].buffer()->address();

    // Fresh fused-operand / chunk addresses (cache replay with new buffers). Layout mirrors create()'s
    // appended writer args (index 17+): [bias] [residual gate bcast] [n_chunks out_ntc chunk1..].
    const uint32_t bias_addr = sv.has_bias ? tensor_args.bias_tensor->buffer()->address() : 0u;
    const uint32_t ta_addr = sv.has_ternary ? tensor_args.fused_ternary_input_a->buffer()->address() : 0u;
    const uint32_t tb_addr = sv.has_ternary ? tensor_args.fused_ternary_input_b->buffer()->address() : 0u;

    // Some configs place every core on a single NoC group (e.g. preaders==1 => all noc 0), leaving the
    // other group's kernel handles unset. Only fetch runtime-arg maps for groups that actually exist.
    bool has_g0 = false, has_g1 = false;
    for (const auto n : sv.core_noc) {
        (n ? has_g1 : has_g0) = true;
    }
    auto* readerA_args = has_g0 ? &GetRuntimeArgs(program, sv.readerA) : nullptr;
    auto* readerB_args = has_g1 ? &GetRuntimeArgs(program, sv.readerB) : nullptr;
    auto* writerA_args = has_g0 ? &GetRuntimeArgs(program, sv.writerA) : nullptr;
    auto* writerB_args = has_g1 ? &GetRuntimeArgs(program, sv.writerB) : nullptr;

    for (uint32_t i = 0; i < sv.num_cores; ++i) {
        const CoreCoord& core = sv.cores[i];
        const bool b = sv.core_noc[i] != 0u;

        // reader arg 0 = in1_addr.
        auto& ra = (*(b ? readerB_args : readerA_args))[core.x][core.y];
        ra[0] = in1_addr;

        // writer arg 0 = in0_addr, arg 1 = out_addr (chunk 0).
        auto& wa = (*(b ? writerB_args : writerA_args))[core.x][core.y];
        wa[0] = in0_addr;
        wa[1] = out_addr;
        uint32_t fidx = 17u;
        if (sv.has_bias) {
            wa[fidx++] = bias_addr;
        }
        if (sv.has_ternary) {
            wa[fidx++] = ta_addr;
            wa[fidx++] = tb_addr;
            fidx++;  // broadcast_gate flag is shape-derived, unchanged across replays
        }
        if (sv.n_chunks > 1u) {
            fidx += 2u;  // n_chunks, out_ntc (unchanged)
            for (uint32_t c = 1; c < sv.n_chunks; ++c) {
                wa[fidx++] = tensor_return_value[c].buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
