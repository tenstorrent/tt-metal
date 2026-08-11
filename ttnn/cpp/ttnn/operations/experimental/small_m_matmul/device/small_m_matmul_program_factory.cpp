// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "small_m_matmul_program_factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/device.hpp>  // get_worker_noc_hop_distance (M-split placement + ring order)

#include "small_m_matmul_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/kernels/in1_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/kernels/compute.cpp";

// Tile-byte sizes are defined once in small_m_matmul_plan.hpp (single source of truth), reached via `plan::`.
using plan::kTileBytesBf16;
using plan::kTileBytesFp32;

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

// M-split (Sm>1) worker PLACEMENT (IN1_NEAR). Overrides ONLY P.cores[i].coord; logical core indices,
// ownership, and the factory's reader->i+s / slave->i-mm runtime-arg math are unchanged. MUST run BEFORE the
// ring reorder so the ring order recomputes on the new coords. Pass 1 places every mm==0 DRAM reader around
// its bank target (a logical-Manhattan spiral mirroring the planner's find_near) so slaves can't displace
// later readers from bank-adjacent cores; pass 2 places each slave at the free worker minimizing the directed
// reader->slave hop on the group's in1-reader NoC. No-op / never called at Sm==1.
void place_m_split_workers(plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo) {
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
    // pass 2: slaves — IN1_NEAR minimizes the directed reader->slave hop on the reader NoC.
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (P.cores[i].mm == 0u) {
                continue;
            }
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
        }
    }
}

// Physical-topology-aware in0 ring ordering (PARETO). Overrides ring_pos/ring_next_idx/ring_prev_idx per ring
// group using the group's WRITER NoC authoritative hop distance (get_worker_noc_hop_distance; logical->physical
// + directed torus routing w/ wraparound). Placement / work / reduction are unchanged; only the ring visiting
// order (which core seeds which in0 shard, the forward route, the in1 rotated read) changes — correct for ANY
// permutation.
//
// M-split (Sm>1): slices differing only in mm form a (kk,nn) group of Sm CONTIGUOUS slice indices [base,
// base+Sm), all sharing the same writer NoC. Their in1 slaves receive in1 in the mm==0 READER's shard order
// while their in0 rings are separate physical cores, so the WHOLE group MUST use the SAME permutation
// (reader/slave ring_pos must agree per bank) or the in0/in1 pairing corrupts.
//
// PARETO objective (aggregated over the Sm physical mm-rings; aggmax = worst directed edge over all rings,
// aggtot = summed hops over all rings): min aggmax subject to aggtot <= the MM0 order's aggtot, then aggtot.
// MM0 (score only the mm==0 ring: min ring0.max then ring0.total) establishes the aggtot budget and seeds the
// search, so PARETO route-dominates MM0 by construction (never a worse total) — it keeps the Sm=2 win and
// stays within noise of MM0 on Sm=1.
void optimize_in0_ring_order(plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo, uint32_t Sm) {
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
        // rings; aggtot = summed hops over all rings.
        struct Metrics {
            uint32_t r0max, r0tot, aggmax, aggtot;
        };
        auto metrics = [&](const std::array<uint32_t, 8>& ord) -> Metrics {
            Metrics m{0, 0, 0, 0};
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                const auto [rm, rt] = ring_cost(ord, dm[mm]);
                if (mm == 0) {
                    m.r0max = rm;
                    m.r0tot = rt;
                }
                m.aggmax = std::max(m.aggmax, rm);
                m.aggtot += rt;
            }
            return m;
        };
        auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) { return a0 < b0 || (a0 == b0 && a1 < b1); };
        auto cand_of = [](const std::array<uint32_t, 7>& t) {
            std::array<uint32_t, 8> c{};
            c[0] = 0;
            for (uint32_t i = 0; i < 7u; ++i) {
                c[i + 1u] = t[i];
            }
            return c;
        };
        const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
        // exhaustive: fix bank 0 at pos 0, permute the other 7 (5040 cycles; directed => both orientations).
        // Pass 1 — MM0 objective (establishes the PARETO aggtot budget).
        std::array<uint32_t, 8> opt_mm0 = bank;
        Metrics b_mm0{~0u, ~0u, ~0u, ~0u};
        std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
        do {
            const std::array<uint32_t, 8> cand = cand_of(tail);
            const Metrics m = metrics(cand);
            if (lt2(m.r0max, m.r0tot, b_mm0.r0max, b_mm0.r0tot)) {
                b_mm0 = m;
                opt_mm0 = cand;
            }
        } while (std::next_permutation(tail.begin(), tail.end()));
        // Pass 2 — PARETO: min aggmax (then aggtot) subject to aggtot <= MM0's aggtot. Seeded with MM0 itself
        // (satisfies the constraint by construction).
        std::array<uint32_t, 8> opt_pareto = opt_mm0;
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
        // apply the PARETO order to ALL Sm slices of this group (same permutation => reader/slave ring_pos
        // agree per bank, preserving in0/in1 pairing under M-split).
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            const uint32_t jj = base + mm;
            for (uint32_t pos = 0; pos < 8u; ++pos) {
                const uint32_t ci = opt_pareto[pos] * preaders + jj;
                P.cores[ci].ring_pos = pos;
                P.cores[ci].ring_next_idx = opt_pareto[(pos + 1u) % 8u] * preaders + jj;
                P.cores[ci].ring_prev_idx = opt_pareto[(pos + 7u) % 8u] * preaders + jj;
            }
        }
    }
}

// TEST-ONLY (diag bit13 PLACE_MESH): 2D (bank x slice) MESH placement, aimed at in0 ring traffic.
// Host-only and correctness-preserving - writes only P.cores[i].coord.
//
// The structural point: two traffic classes want opposite groupings of the same 8 x preaders core array.
// The in0 RING connects the 8 cores of one SLICE (one per bank), so it wants slice-compact clusters. The
// split-K REDUCTION chain connects the Pk cores of one BANK, so it wants bank-compact clusters. Those are
// orthogonal partitions, so no clustering makes both short - production picks bank-compact blobs (short
// reduction, long ring). A 2D EMBEDDING escapes the tension: put banks along x and slices along y, and then a
// ring step (bank -> bank) and a reduction step (kk -> kk+1) are each ONE hop, in different dimensions.
//
// Offline (in0_ring_place_search.py, exact route model): ring hops -70% AND reduction hops -19..-40%
// simultaneously, whole-op peak link load -11..-15%, total link traffic -20..-36%. in1 read distance is
// ~unchanged (+3%), which is acceptable because the in1 read was measured to be DRAM-bound (76-98% of peak
// in isolation) and insensitive to distance - see IN1_PLACEMENT_AB.md.
//
// Layout: cores (bank b, slice p) -> (x=b, y=p) for p < grid.y; overflow slices each take their own column at
// x >= 8 with the 8 banks down rows 0..7. Collision-free by construction. mm-siblings are consecutive in p,
// so M-split slaves land adjacent to their reader, which keeps the in1 forward short without a separate pass.
void place_mesh(plan::ExecutionPlan& P, const plan::Geometry& geo, const CoreCoord& grid) {
    const uint32_t preaders = geo.num_cores / 8u;
    const uint32_t overflow_cols = (grid.x > 8u) ? (grid.x - 8u) : 0u;
    TT_FATAL(
        grid.x >= 8u && preaders <= grid.y + overflow_cols,
        "small_m_matmul mesh placement does not fit: {} slices need <= {} (grid {}x{})",
        preaders,
        grid.y + overflow_cols,
        grid.x,
        grid.y);
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (p < grid.y) {
                P.cores[i].coord.x = b;  // banks along x, slices along y
                P.cores[i].coord.y = p;
            } else {
                P.cores[i].coord.x = 8u + (p - grid.y);  // one spare column per overflow slice
                P.cores[i].coord.y = b;
            }
        }
    }
}

}  // namespace

SmallMMatmulProgramFactory::cached_program_t SmallMMatmulProgramFactory::create(
    const SmallMMatmulParams& operation_attributes,
    const SmallMMatmulInputs& tensor_args,
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
    const SmallMMatmulConfig cfg = operation_attributes.config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));

    // ---- Fused epilogue detection (all off => byte-identical no-fusion path). ----
    // Detected BEFORE planning: the fused operand CBs (c_4..c_6) and the reduce-scatter epilogue scratch
    // (c_10) are real L1, so the planner's feasibility check needs them to be authoritative.
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
    const plan::FusionInputs fusion{
        .has_bias = has_bias,
        .has_ternary = has_ternary,
        .has_activation = has_activation,
        // broadcast_gate defaults to 1 when there is no ternary; only meaningful when has_ternary.
        .broadcast_gate = has_ternary && broadcast_gate != 0u,
        .gate_is_fp32 = gate_is_fp32};

    // ---- Run the pure host planner ----
    auto planres = make_and_build_plan(device, in0, in1, cfg, fusion);
    TT_FATAL(planres.ok(), "small_m_matmul planner rejected config: {}", planres.error);
    plan::ExecutionPlan& P = *planres.plan;  // mutable: the ring-order diag overrides ring_pos/next/prev below
    const plan::Geometry& geo = P.geo;
    const plan::CbSizes& cb = P.cb;

    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    // `use_reduce` doubles as the reduction-CB DEPTH: 0 when Pk==1 (no chain), else the number of cb7 slots.
    // The kernel takes its slot modulus from this same value, so the CB size and the remote write offset can
    // never disagree - the failure mode that produced a PCC 0.38 bug in earlier reduction work.
    // It is deliberately NOT derived from cb.cb7_tiles: that is 0 under reduce-scatter (which allocates
    // c_8/c_9 instead), and a 0 here reads to the kernels as "Pk == 1, no reduction at all", so the writer
    // takes the no-reduce exit while compute still waits for ring partials -> hang.
    const uint32_t use_reduce = (Pk > 1u) ? plan::kCb7Depth : 0u;

    // ---- Output-split detection ----
    const int32_t chunks = operation_attributes.chunks < 1 ? 1 : operation_attributes.chunks;
    const uint32_t n_chunks = static_cast<uint32_t>(chunks);
    const uint32_t out_ntc = Nt_r / n_chunks;  // per-chunk N tiles (validated divisible + tile-aligned)

    // ---- Kernel compile defines. wdefs = writer (in0 ring/reduce + fused output); fdefs_compute (below) =
    // compute fusion defines merged into cdefs. The in1 reader takes NO defines. Empty maps => the
    // byte-identical no-fusion compile. ----
    std::map<std::string, std::string> wdefs;
    // extra COMPUTE defines beyond fusion; currently only the reduction strategy (RSCATTER).
    std::map<std::string, std::string> cdefs_extra;

    // ---- INTERNAL REDUCTION STRATEGY: linear chain (default) vs ring REDUCE-SCATTER. ----
    // The chain sends each of the Pk-1 non-root bands' FULL output block one hop up, so the last partial only
    // starts moving after Pk-2 earlier hops and the root alone writes all the output. Reduce-scatter instead
    // tile-partitions the block into Pk chunks and rotates them around the Pk cores: the same total number of
    // adds and the same total bytes, but every core sends concurrently every round, and each core ends up
    // owning + writing ONE fully-reduced chunk, so the output write is spread over Pk cores instead of 1.
    //
    // Adopted only in the case where it was MEASURED to win 5-9% with zero regressions (five corpus shapes:
    // 64/128/256 x 2048 x 1024/2048): Pk>=4, shallow K (the deep-K shapes are in1-read-bound, so the reduction
    // is not on the critical path), wide enough N, N_sub>=2, and a block that partitions evenly over Pk.
    // Fusion and chunked output ARE supported: each owner applies the epilogue exactly once to its own fully
    // reduced slice, and the writer feeds that owner only its slice's operands, in slice order, so compute
    // indexes them 0..nt-1. The ring's send/recv CBs live at c_8/c_9 so they cannot collide with the fusion
    // operand CBs c_4/c_5/c_6.
    //
    // NOTE: reduce-scatter re-associates the K-sum (each owner adds the Pk partials in ring order instead of
    // bottom-to-top), so it is PCC-preserving but NOT bit-identical to the chain. Every add stays in FP32 and
    // no operand narrows, so this is a different summation ORDER, not a precision reduction.
    //
    // bit22 = FORCE_CHAIN (A/B the chain baseline), bit23 = FORCE_RSCATTER (test it outside the gate). Any
    // kernel-behaviour diagnostic (ablations, meet) keeps the chain so the ablation floors stay comparable.
    // The partition does NOT need to divide evenly: chunk sizes differ by at most one tile, so the only
    // structural requirement is rs_T >= Pk (every core must own at least one tile to write). Requiring
    // divisibility locked out 41 of the 62 corpus shapes for no reason.
    const uint32_t rs_T = geo.M_block_capacity * geo.N_sub;  // tiles per output sub-block
    // No N-WIDTH requirement. The original gate also demanded Nt>=32; measuring the declined shapes with bit23
    // showed that was wrong - 128x2048x512 (Nt=16) is 8.9% FASTER with reduce-scatter. Every shallow-K Pk>=4
    // shape with N_sub>=2 won (6/6, 5.5-14.7%) regardless of N width, and N_sub==1 was neutral (+0.4%).
    //
    // DEEP K is admitted only under two extra conditions, because reduce-scatter trades data movement for
    // per-round compute SETUP cost (each round pays an add_tiles_init + data-format reconfig no matter how few
    // tiles it touches). It therefore needs FEW rounds and ENOUGH WORK per round:
    //   Pk <= 6        -> at most 5 rounds per sub-block. At Pk=12 the 11 rounds cost far more than the
    //                     shortened critical path saves: 512x6144x4608 +33.4%, 512x6144x2304 +19.5%,
    //                     128x15360x1536 +3.0%.
    //   max_chunk >= 2 -> a round that moves a single 2 KB tile is almost pure overhead: 64x15360x1536
    //                     (Pk=6, 1-tile chunks) is +2.7% despite satisfying the round bound.
    // Together these separate all 13 measured deep-K shapes exactly: the 6 satisfying both won (-1.3% to -3.8%)
    // and all 5 losses are excluded. Shallow K still wins with single-tile chunks (compute has far more slack
    // there - 30-40% of the wall vs 47-77%), so the chunk-size floor is deep-K only.
    // The gate itself lives in plan::rscatter_selected() so that the L1 accounting charges for exactly the
    // buffers this decision allocates. cb.rscatter is that same call, made during planning; assert they agree
    // rather than trusting it, since a divergence would silently under-charge L1.
    const bool rscatter = plan::rscatter_selected(Pk, Kt_r, geo.M_block_capacity, geo.N_sub);
    TT_FATAL(
        rscatter == cb.rscatter,
        "internal: reduction strategy disagrees with the L1 accounting (factory={}, planner={})",
        rscatter,
        cb.rscatter);
    if (rscatter) {
        wdefs["RSCATTER"] = "1";
        cdefs_extra["RSCATTER"] = "1";
    }

    // ---- M-split worker PLACEMENT (Sm>1): IN1_NEAR. Overrides only P.cores[i].coord; MUST run BEFORE the ring
    // reorder so the ring order recomputes on the placed coords. No-op at Sm==1. ----
    // PRODUCTION GATE for the 2D mesh placement. Adopt only when the mesh FILLS the grid: preaders >= 10
    // puts at least one slice in every grid row, so cores spread over the whole 11x10 array. Below that the
    // mesh packs all 8*preaders cores into rows 0..preaders-1 and concentrates every DRAM path into a corner -
    // measured -48% to -89% at preaders<=5 and -6% to -23% at preaders 6..7. Sm>1 is excluded because M-split
    // slaves then lose the IN1_NEAR pass and the measured results are mixed (+8.7% to -22.2%), and Pk>=4 is
    // required for the Ns>1 case (Pk=3/Ns=4 measured neutral-to-negative).
    // Fitted on the 63-shape corpus at deployed configs: adopts 24/63 shapes, mean +5.06%, best +14.98%,
    // worst -1.27%; every declined shape stays byte-identical to the previous production placement.
    // Second clause: adopt regardless of the above when the in0 RING simply carries more traffic than the in1
    // read - at ring >= 2x in1 the ring savings dominate whatever the placement costs the read. On the corpus
    // exactly 3 shapes clear 2x and all 3 win (+14.64%, +9.09%, +8.70%); the highest-ratio loser is at 1.31x.
    // This is what lets the two Sm>1 ring-heavy shapes in (256x15360x768, 256x2048x512).
    const uint32_t Ns_gate = cfg.n_slices ? cfg.n_slices : 1u;
    const uint64_t ring_bytes =
        static_cast<uint64_t>(geo.num_cores) * 7u * geo.W * geo.M_block_capacity * kb * kTileBytesBf16;
    const uint64_t in1_bytes = static_cast<uint64_t>(geo.Kt) * geo.Nt * kTileBytesBf16;
    // Mt >= 8 is REQUIRED. The mesh trades in1 read locality (cores leave their own DRAM bank) for a ~70% cut
    // in in0 ring traffic, and ring traffic per shard scales with M_block = Mt/Sm. At Mt <= 4 the shards are
    // so small that there is almost no ring traffic to save while the read penalty is paid in full: measured
    // 9.6-13.1% SLOWER on 32x6080x4640, 128x6144x4608, 64x6144x768, 128x6144x2304, 64x15360x768 and
    // 128x15360x1536 (all Pk=12, i.e. inside the old gate). At Mt >= 8 it is 16-24% FASTER on the same
    // topology (512x6144x2304 +16.4%, 256x6144x768 +23.9%).
    const bool mesh_gate = geo.Mt >= 8u && (((Pk * Ns_gate >= 10u) && (Sm == 1u) && (Ns_gate == 1u || Pk >= 4u)) ||
                                            (ring_bytes >= 2u * in1_bytes));
    const bool use_mesh = mesh_gate;
    // OBSERVABILITY ONLY (TT_SMALL_M_LOG_CFG): report what the picker and the internal gates actually chose.
    // Runs once per program-cache miss and changes NOTHING about behaviour -- there is no way to read the
    // auto-selected config from Python otherwise, and reporting a host-side mirror of the picker risks silently
    // misreporting if the mirror drifts from auto_select_config.
    if (std::getenv("TT_SMALL_M_LOG_CFG") != nullptr) {
        log_info(
            tt::LogOp,
            "small_m_cfg M={} K={} N={} pick=({},{},{},{},{}) cores={} reduction={} placement={}",
            Mt_r * 32u,
            Kt_r * 32u,
            Nt_r * 32u,
            Pk,
            cfg.n_slices ? cfg.n_slices : 1u,
            Sm,
            kb,
            cfg.n_subblock_tiles,
            geo.num_cores,
            rscatter ? "reduce-scatter" : "chain",
            use_mesh ? "mesh" : (Sm > 1u ? "in1-near" : "bank-local"));
    }
    if (use_mesh) {
        place_mesh(P, geo, device->compute_with_storage_grid_size());
    } else if (Sm > 1u) {
        place_m_split_workers(P, device, geo);
    }

    // ---- Physical-topology-aware in0 ring ordering (PARETO) over each (kk,nn) group's Sm mm-rings. ----
    optimize_in0_ring_order(P, device, geo, Sm);

    // ---- REDUCE-SCATTER cyclic order over each group's Pk cores (runs AFTER placement + ring ordering, so it
    // sees the final coords). For each (bank b, within-bank sub) group, order the Pk k-slice cores into a
    // Hamiltonian CYCLE minimizing the worst DIRECTED NoC hop: one bad wraparound edge would serialize every
    // round of the ring, so the max edge matters more than the total. The edge cost a->c is measured on the
    // SENDER's writer NoC (writer runs opposite the core's in1-reader NoC), which is asymmetric on the torus
    // and therefore cannot be approximated by a coordinate distance. Pk==4 searches all 3! orders exactly;
    // larger Pk uses greedy nearest-neighbour (P! is infeasible). Mutates only the rs_* fields. ----
    if (rscatter) {
        namespace expdev = tt::tt_metal::experimental::Device;
        for (uint32_t b = 0; b < 8u; ++b) {
            for (uint32_t sub = 0; sub < geo.mfac; ++sub) {
                std::vector<uint32_t> idx(Pk);
                for (uint32_t kk = 0; kk < Pk; ++kk) {
                    idx[kk] = b * geo.preaders + kk * geo.mfac + sub;
                }
                auto lc = [&](uint32_t li) {
                    const auto& c = P.cores[idx[li]].coord;
                    return CoreCoord{c.x, c.y};
                };
                auto dist = [&](uint32_t a, uint32_t c) -> uint32_t {
                    if (a == c) {
                        return 0u;
                    }
                    const NOC wnoc = (P.cores[idx[a]].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
                    return expdev::get_worker_noc_hop_distance(device, lc(a), lc(c), wnoc);
                };
                std::vector<uint32_t> ord(Pk);
                if (Pk == 4u) {
                    // exact min-(maxedge, total) cycle over the 3! orderings of {1,2,3} (position 0 fixed)
                    const uint32_t perms[6][3] = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}};
                    uint32_t best_max = ~0u, best_tot = ~0u;
                    for (const auto& pm : perms) {
                        const uint32_t o[4] = {0u, pm[0], pm[1], pm[2]};
                        uint32_t mx = 0, tot = 0;
                        for (uint32_t p = 0; p < 4u; ++p) {
                            const uint32_t e = dist(o[p], o[(p + 1u) % 4u]);
                            mx = std::max(mx, e);
                            tot += e;
                        }
                        if (mx < best_max || (mx == best_max && tot < best_tot)) {
                            best_max = mx;
                            best_tot = tot;
                            for (uint32_t p = 0; p < 4u; ++p) {
                                ord[p] = o[p];
                            }
                        }
                    }
                } else {
                    std::vector<bool> vis(Pk, false);
                    ord[0] = 0u;
                    vis[0] = true;
                    for (uint32_t p = 1; p < Pk; ++p) {
                        uint32_t best = 0, bestd = ~0u;
                        for (uint32_t cand = 0; cand < Pk; ++cand) {
                            if (vis[cand]) {
                                continue;
                            }
                            const uint32_t dd = dist(ord[p - 1], cand);
                            if (dd < bestd) {
                                bestd = dd;
                                best = cand;
                            }
                        }
                        ord[p] = best;
                        vis[best] = true;
                    }
                }
                for (uint32_t p = 0; p < Pk; ++p) {
                    auto& cp = P.cores[idx[ord[p]]];
                    cp.rs_pos = p;
                    cp.rs_next_idx = idx[ord[(p + 1u) % Pk]];
                    cp.rs_prev_idx = idx[ord[(p + Pk - 1u) % Pk]];
                    cp.rs_own_chunk = (p + 1u) % Pk;
                }
            }
        }
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
    // cb7 is the CHAIN's running-sum buffer. Reduce-scatter never touches it (its partials travel through the
    // c_4/c_5 chunk CBs instead), so don't spend the L1 on it there.
    // cb7_tiles is already 0 under reduce-scatter (the sizer decides that), so this one test covers both
    // "Pk == 1, no chain" and "reduce-scatter, partials travel via c_8/c_9 instead".
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // reduce (Pk>1)
    }
    // Fused-epilogue operand CBs (only when the matching fusion is active). c_4 bias [1,N_sub], c_5 residual
    // [M,N] block, c_6 gate [1,N_sub] (broadcast) or [M,N] block. Sized to hold a full sub-block so the
    // writer can stream all M rows while compute consumes them (matches minimal_matmul's ternary CB sizing).
    // Sizes come from the planner's CbSizes (plan::compute_cb_sizes), never recomputed here: what feasibility
    // charged and what we allocate are then the same numbers by construction.
    if (has_bias) {
        mkcb(program, all_cores, 4, cb.cb4_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
    if (has_ternary) {
        mkcb(program, all_cores, 5, cb.cb5_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        const tt::DataFormat gfmt = gate_is_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t gtsz = gate_is_fp32 ? kTileBytesFp32 : kTileBytesBf16;
        mkcb(program, all_cores, 6, cb.cb6_tiles, gfmt, gtsz);
    }
    // REDUCE-SCATTER ring CBs at c_8/c_9 -- deliberately NOT c_4/c_5, which are fusion operands, so the ring
    // and the fused epilogue coexist. EXACTLY 2 slots each, sized to the LARGEST slice
    // (chunks differ by at most one tile when Pk does not divide the sub-block): the kernel's slot index is the
    // global epoch mod 2 and every CB operation moves a whole max-size slot, so the FIFO period and the remote
    // write stride are the same value by construction.
    if (rscatter) {
        mkcb(program, all_cores, 8, cb.cb8_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        mkcb(program, all_cores, 9, cb.cb9_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        if (cb.cb10_tiles > 0u) {
            // c_10: scratch for the reduced slice while the fused epilogue is applied to it in place.
            mkcb(program, all_cores, 10, cb.cb10_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
        }
    }

    // ---- Semaphores ----
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);      // in0 ring recv
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);      // reduction recv (shared counter)
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);  // cb_reduce reverse credit
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;                         // M-split reader<->slaves
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
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
        in1ready_sem};           // 8

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
    // in1 reader takes no compile defines.
    const std::map<std::string, std::string> rdefs;
    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, rdefs);
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, rdefs);
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // compute (spec §6c). fp32 DST limit: subblock_h * subblock_w <= 4.
    // Subblock geometry. fp32_dest_acc_en gives 4 DST tiles, so the hard limit is sbh*sbw <= 4 (exceeding it
    // silently corrupts output - there is a known precedent, hence the assert below). The historical sizer
    // caps subblock_h at 2, so when N_sub == 1 it yields 2x1 = 2 tiles and leaves HALF the DST idle. We now
    // ENLARGE such subblocks to the biggest area that still fits (4x1 when N_sub==1 and M_block%4==0), which
    // halves the matmul_block call count for identical math - verified BIT-EXACT. Measured on the 63-shape
    // corpus: +2.56% on 512x6144x4608, +2.25% on 512x6144x2304, and within noise everywhere else.
    // diag bit16 restores the legacy sizer for A/B.
    uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
    // Only ever ENLARGE the subblock: where the historical sizer already reaches the 4-tile limit, keep its
    // exact shape. Re-shaping an already-maximal subblock (e.g. 2x2 -> 1x4) is not free - it measured -2.22%
    // on 256x2048x1024 - so the rule is a strict Pareto improvement by construction.
    if (sbh * sbw < 4u) {
        uint32_t bh = sbh, bw = sbw;
        for (uint32_t h = 1u; h <= 4u; ++h) {
            if (geo.M_block_capacity % h != 0u) {
                continue;
            }
            for (uint32_t w = 1u; h * w <= 4u; ++w) {
                if (geo.N_sub % w != 0u) {
                    continue;
                }
                if (h * w > bh * bw || (h * w == bh * bw && w > bw)) {
                    bh = h;
                    bw = w;
                }
            }
        }
        sbh = bh;
        sbw = bw;
    }
    TT_FATAL(sbh * sbw <= 4u, "small_m_matmul subblock {}x{} exceeds the 4-tile fp32 DST limit", sbh, sbw);
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
    cdefs.insert(fdefs_compute.begin(), fdefs_compute.end());  // fusion defines (empty for the no-fusion path)
    cdefs.insert(cdefs_extra.begin(), cdefs_extra.end());      // reduction strategy (RSCATTER when gated in)
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
        // Fused-epilogue / output-split writer args (index 17+). Order MUST match the writer kernel's fidx
        // reads: bias, then residual/gate/broadcast, then chunk count/width/addresses.
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
        // REDUCE-SCATTER writer args (index 17+; unfused + single-chunk only, so index 17 is free here).
        if (rscatter) {
            const auto rn = phys(cp.rs_next_idx);
            const auto rp = phys(cp.rs_prev_idx);
            wa.push_back(rn.x);             // 17 next core in the Pk cycle (I send to it)
            wa.push_back(rn.y);             // 18
            wa.push_back(rp.x);             // 19 prev core (it sends to me)
            wa.push_back(rp.y);             // 20
            wa.push_back(cp.rs_own_chunk);  // 21 tile-chunk index this core owns + writes
            wa.push_back(Pk);               // 22 cycle size
            wa.push_back(rs_T);             // 23 sub-block tiles (kernel derives the chunk sizes)
        }
        SetRuntimeArgs(program, wh, cores[i], wa);

        // compute runtime args: fixed rectangular block over the schedule capacities. N_end spans ALL
        // N_bpc sub-blocks (spec §7); zero-filled tail positions contribute zero. When a fusion is active the
        // reduction-root flag (is_top) follows, then the addcmul scalar bits + gate-broadcast flag.
        std::vector<uint32_t> ca = {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u};
        if (rscatter) {
            ca.push_back(cp.rs_pos);  // my position in the Pk cycle
            ca.push_back(Pk);         // cycle size
            ca.push_back(rs_T);       // sub-block tiles (kernel derives the chunk sizes)
        }
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

void SmallMMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SmallMMatmulParams& /*operation_attributes*/,
    const SmallMMatmulInputs& tensor_args,
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
