// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/device.hpp>         // get_worker_noc_hop_distance (M-split placement + ring order)
#include <tt-metalium/mesh_device.hpp>                 // MeshDevice (resolve a unit device for the NoC queries)
#include <tt-metalium/experimental/fabric/fabric.hpp>  // FabricMuxV2Config, add_fabric_mux_v2_to_program
#include "ttnn/operations/ccl/ccl_common.hpp"          // rank / neighbour resolution along a cluster axis

#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in1_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/compute.cpp";

// Tile-byte sizes are defined once in ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp
// (single source of truth), reached via `plan::`.
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
        "all_gather_regime_a_matmul_async mesh placement does not fit: {} slices need <= {} (grid {}x{})",
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

// ---------------------------------------------------------------------------------------------------
// PHASE 2 direct-L1 streaming: per-core stream plan.
// ---------------------------------------------------------------------------------------------------
// Every consumer core is its own fabric client. Core (kk, ring_pos) on device d receives its cb0 slot 0
// straight into L1 from the SAME core index on the upstream device and relays those same bytes once to the
// downstream one, so relay source == consume destination: no relay buffer, no extra L1, and no credit or
// bounded-window protocol (slot 0 is written exactly once and nothing in the program reuses it). See
// tools/mm_sweep/AGMM_DIRECT_L1_DESIGN.md, "Dataflow".
//
// The whole scheme rests on one property: slot 0 of core (kk, p) must hold exactly ONE source rank's tiles,
// so that one origin can fill it with one contiguous transfer. That is NOT automatic -- see rank_span.
struct DirectL1Core {
    uint32_t src_rank = 0;     // the rank whose shard fills this core's cb0 slot 0
    uint32_t dist = 0;         // hops from src_rank along this core's stream (0 == origin: reads local DRAM)
    uint32_t run_len = 0;      // VALID K tiles per (rank, Pk group) stripe
    uint32_t stripe_base = 0;  // this Pk group's first K tile within a source shard
    uint32_t rank_span = 0;    // capacity-local K slots reserved per rank (== pos_per_rank * W*kb)
    bool has_stripe = false;   // false => this ring position is pure zero padding: no fabric, no DRAM read
    bool send_fwd = false;     // originates/relays toward rank+1
    bool send_bwd = false;     // ... toward rank-1
};

// RANK-ALIGNED BLOCKED-CYCLIC K -- why direct-L1 needs its own K mapping.
//
// The DRAM-staged path packs each Pk group's per-shard stripes back to back, so capacity-local index l
// belongs to source rank l / run_len. Ring slot boundaries, however, sit at multiples of W*kb. Unless
// run_len happens to be a multiple of W*kb a single slot 0 STRADDLES two ranks, and a straddling slot
// cannot be filled by one contiguous transfer from one origin. This is not a corner case: `medium` at Pk=4,
// kb=2 gives run_len=10 against W*kb=6.
//
// So here each rank is given a whole number of ring positions: pos_per_rank = cdiv(run_len, W*kb), i.e.
// rank_span = pos_per_rank * W*kb capacity-local slots of which the first run_len are valid and the rest
// zero. Since tp * pos_per_rank <= 8, the total tp * rank_span <= 8 * W*kb == K_slice_capacity: this spends
// only capacity the staged path was ALREADY leaving as its k-tail. It redistributes the zero padding (from
// one run at the end to a little after each rank), it does not add any, and compute cost is unchanged
// because the kernels always walk the full capacity and zero-fill invalid positions either way.
//
// Both the in0 side and the in1 reader take rank_span as a runtime arg and evaluate the same gk(l), so the
// two stay in lockstep -- the spec requires the in1 reader to walk the identical global-K order. Passing
// rank_span == run_len (what the staged path does) reduces every formula back to the old one exactly.
std::vector<DirectL1Core> build_direct_l1_plan(
    const plan::ExecutionPlan& P,
    const plan::Geometry& geo,
    uint32_t Pk,
    uint32_t kb,
    uint32_t tp,
    uint32_t rank,
    bool topology_is_ring,
    // Balanced bidirectional delivery (spec appendix A): each stripe leaves its origin BOTH ways, so it
    // travels tp/2 hops instead of tp-1. Implemented and correct, but OFF by default because it MEASURES
    // WORSE on the dependent path -- see the table at the call site.
    bool balanced) {
    std::vector<DirectL1Core> out(geo.num_cores);
    const uint32_t Wkb = geo.W * kb;  // capacity-local K tiles per ring slot
    const uint32_t k_shard_tiles = geo.Kt / tp;
    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const plan::CorePlan& cp = P.cores[i];
        DirectL1Core d;
        const plan::BalRange rs = plan::rap_balanced(cp.kk, k_shard_tiles, Pk);
        d.run_len = rs.extent;
        d.stripe_base = rs.start;
        const uint32_t pos_per_rank = (d.run_len + Wkb - 1u) / Wkb;
        d.rank_span = pos_per_rank * Wkb;
        // tp * pos_per_rank <= 8 is what makes the tp ranks fit in the 8 ring positions. It follows from
        // K_slice_capacity >= ceil(Kt/Pk) for every shape this op accepts, but a violation would silently
        // alias two ranks onto one slot, so check rather than assume.
        TT_FATAL(
            tp * pos_per_rank <= 8u,
            "TT_AGMM_DIRECT_L1: Pk group {} needs {} ring positions per source rank (run_len={} over a "
            "W*kb={} slot), which does not fit tp={} ranks in 8 positions. Unset TT_AGMM_DIRECT_L1 to use "
            "the DRAM-staged path.",
            cp.kk,
            pos_per_rank,
            d.run_len,
            Wkb,
            tp);
        d.has_stripe = cp.ring_pos < tp * pos_per_rank;
        if (d.has_stripe) {
            d.src_rank = cp.ring_pos / pos_per_rank;
            if (topology_is_ring && !balanced) {
                // DEFAULT: one direction per ring position, by parity. A stripe travels from its origin all
                // the way round, tp-1 hops. Deeper than appendix A's schedule, but measured FASTER on the
                // dependent path -- see the table where `balanced` is resolved.
                const bool bwd = (cp.ring_pos % 2u) != 0u;
                d.dist = bwd ? ((d.src_rank + tp - rank) % tp) : ((rank + tp - d.src_rank) % tp);
                const bool relays = (d.dist + 1u) < tp;  // the last hop consumes without forwarding
                d.send_fwd = !bwd && relays;
                d.send_bwd = bwd && relays;
            } else if (topology_is_ring) {
                // BALANCED BIDIRECTIONAL delivery -- spec appendix A. Each stripe leaves its origin in BOTH
                // directions, so it travels at most tp/2 hops instead of all the way round. Same bytes, but
                // T_ready_max (the leading term of the ring bound) drops with the hop depth: tp=4 3 -> 2,
                // tp=8 7 -> 4.
                //
                // The antipode (distance exactly tp/2 at even tp) is equidistant both ways, so one direction
                // must carry it -- and if it were always the same one, that link would move tp/2 * K/tp
                // against the other's (tp/2-1) * K/tp (2x at tp=4). Split it at STRIPE granularity by giving
                // each stripe's extra hop to alternating directions. Never at byte granularity: a core keeps
                // receiving exactly ONE stripe with ONE credit, so there is no partial slot to reconcile.
                //
                // The alternation index is the SLICE, not the Pk group: at tp=8 pos_per_rank == 1 leaves no
                // within-rank positions to alternate over, so the slice term carries it alone (and gives
                // preaders distinct values where kk would give only Pk).
                const bool via_fwd = (((cp.ring_pos % pos_per_rank) + cp.slice) % 2u) == 0u;
                const uint32_t f = via_fwd ? (tp / 2u) : (tp / 2u - 1u);
                const uint32_t b = tp - 1u - f;  // f + b == tp-1 always
                const uint32_t fd = (rank + tp - d.src_rank) % tp;
                const uint32_t bd = (d.src_rank + tp - rank) % tp;
                // fd + bd == tp and f + b == tp-1, so for fd != 0 exactly one arrival case holds.
                if (fd == 0u) {
                    d.dist = 0u;  // origin: reads its own shard, then seeds both directions
                    d.send_fwd = f >= 1u;
                    d.send_bwd = b >= 1u;
                } else if (fd <= f) {
                    d.dist = fd;
                    d.send_fwd = fd < f;  // relay onward unless we are this direction's terminal
                } else {
                    // Not reachable unless f + b != tp-1. Checked rather than assumed because the failure is
                    // a HANG, not a wrong answer: this core would wait on an arrival semaphore that no
                    // upstream device ever credits.
                    TT_FATAL(
                        bd <= b,
                        "TT_AGMM_DIRECT_L1: stripe (slice {}, pos {}) from rank {} reaches neither stream on "
                        "rank {} (fd={} > f={}, bd={} > b={}); f+b must equal tp-1={}",
                        cp.slice,
                        cp.ring_pos,
                        d.src_rank,
                        rank,
                        fd,
                        f,
                        bd,
                        b,
                        tp - 1u);
                    d.dist = bd;
                    d.send_bwd = bd < b;
                }
            } else {
                // LINE: there is no wrap, so a stripe has to fan out BOTH ways from its origin -- the origin
                // is the one core that drives two muxes. Everyone else relays outward only.
                if (rank == d.src_rank) {
                    d.dist = 0;
                    d.send_fwd = (rank + 1u) < tp;
                    d.send_bwd = rank > 0u;
                } else if (rank > d.src_rank) {
                    d.dist = rank - d.src_rank;
                    d.send_fwd = (rank + 1u) < tp;
                } else {
                    d.dist = d.src_rank - rank;
                    d.send_bwd = rank > 0u;
                }
            }
        }
        out[i] = d;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------
// PHASE 1 fused fabric all-gather: per-device host context.
// ---------------------------------------------------------------------------------------------------
// Built once per mesh coordinate inside create_at, only when tp > 1. Resolves this rank's position in the
// TP group and its two neighbours, then stands up ONE mux v2 per direction. Only the master in0 ring
// (ring group p == 0, i.e. cores i = b*preaders for b in 0..7) connects to the muxes -- per the design
// discussion, forwarding work is not yet spread across the other rings.
struct FusedGatherContext {
    bool enabled = false;
    uint32_t rank = 0;  // this device's index in the TP group
    uint32_t tp = 1;
    // A direction is absent at the ends of a LINE; on a ring both are always present.
    // One mux per (direction, link). A direction is absent at a line end, in which case its vector is
    // empty. Clients are dealt round-robin across a direction's links.
    std::vector<CoreCoord> mux_virtual_cores_fwd;
    std::vector<CoreCoord> mux_virtual_cores_bwd;
    std::vector<std::unique_ptr<tt::tt_fabric::FabricMuxV2Config>> mux_cfgs_fwd;
    std::vector<std::unique_ptr<tt::tt_fabric::FabricMuxV2Config>> mux_cfgs_bwd;
    std::vector<uint8_t> next_channel_fwd;
    std::vector<uint8_t> next_channel_bwd;
    uint32_t num_links = 1;
    uint32_t channels_per_mux = 0;
    // Per-mux channel counts. Under direct-L1 the client set is the whole compute grid and, on a LINE, its
    // size depends on this device's rank, so the two directions no longer have equal or link-divisible
    // counts. Each mux is therefore sized to exactly the clients dealt to it: mux v2 self-terminates by
    // counting close() calls against its compile-time channel count, so an over-provisioned mux does not
    // error -- its forwarder RISC simply never exits.
    std::vector<uint8_t> mux_channels_fwd;
    std::vector<uint8_t> mux_channels_bwd;
    // Payload bytes per fabric packet on the direct-L1 path, derived from the fabric's own max payload.
    uint32_t dl1_packet_bytes = 0;
    // Kept for the observability log: both are derived, so they are worth being able to see.
    uint32_t mux_depth = 0;
    size_t mux_channel_bytes = 0;
    // Round-robin dealing counters over a direction's links (direct-L1 only; the staged path keys the link
    // off bank_id instead).
    uint32_t next_link_fwd = 0;
    uint32_t next_link_bwd = 0;
    // Readiness: one semaphore slot per (source rank, chunk). Receivers block on these; senders
    // atomic-inc AFTER the payload is flushed.
    uint32_t chunk_ready_sem_id = 0;   // VALID/INVALID go-ahead flag, on EVERY core
    uint32_t gather_count_sem_id = 0;  // masters-done counter, meaningful on master 0
    uint32_t local_done_sem_id = 0;    // local-staging-done counter, held by EVERY master
    // Progressive consumption. Each direction has a COORDINATOR master (bank 0 fwd, bank m_groups bwd).
    // Its peers report each arrival into dir_count_sem on the coordinator; the coordinator then publishes
    // the highest globally-complete arrival index into wave_{fwd,bwd}_sem on EVERY core by multicast.
    // Single writer per wave semaphore and monotone values, so consumers can wait_min on them -- this is
    // the per-slot epoch the design spec asks for, not an ambiguous shared counter.
    uint32_t dir_count_sem_id = 0;
    uint32_t wave_fwd_sem_id = 0;
    uint32_t wave_bwd_sem_id = 0;
    CoreCoord fwd_coord_virtual{};
    CoreCoord bwd_coord_virtual{};
    uint32_t fwd_coord_swaps = 0;
    uint32_t bwd_coord_swaps = 0;
    // Barrier geometry, in VIRTUAL (translated) coords.
    CoreCoord master0_virtual{};
    // Multicast rectangles covering every core master 0 releases: {start_x, start_y, end_x, end_y, dests}.
    struct ReleaseRange {
        // Two destination counts, one per possible sender. A non-loopback multicast must NOT count the
        // sender in num_dests, and there are now two cores that publish: the forward coordinator (master 0)
        // and the backward coordinator (master m_groups). Counting the sender waits on an ack that never
        // arrives, which hangs.
        uint32_t sx, sy, ex, ey, dests_fwd, dests_bwd;
    };
    std::vector<ReleaseRange> release_ranges;
    std::vector<CoreCoord> master_virtuals;  // all masters, for the local-staging barrier
    uint32_t num_masters = 8;
    // Counts forward-stream shard arrivals from the backward neighbour. This is a caller-owned GLOBAL
    // semaphore address, not a program semaphore id: see the note in the operation types header for why a
    // cross-chip credit cannot land in a program semaphore.
    uint32_t fwd_recv_sem_addr = 0;
    uint32_t bwd_recv_sem_addr = 0;  // gather_semaphores[1]; the backward stream's own counter
};

// ---------------------------------------------------------------------------------------------------
// On-chip ring SLOT SCHEDULE: writer args 17..32, part of the FIXED prefix.
// ---------------------------------------------------------------------------------------------------
// A cb0 slot index is a CONSUMPTION-ORDER index, not a physical identity: compute waits cumulatively
// (`cb_wait_front(in0_cb, (k_block+1)*in0_block_num_tiles)`) and addresses each block by an explicit offset
// (`k_block * in0_block_num_tiles`), walking k_block ascending. So "the stripe consumed s-th" and "the stripe
// in slot s" are the same statement, and the only way to change consumption order is to change WHICH STRIPE
// LANDS IN WHICH SLOT.
//
// Today that mapping is implicit in the step counter: the own stripe goes to slot 0, and the forward at step s
// reads slot s and writes slot s+1 on the successor, so core c consumes position (c-s) at step s -- an order
// chained to its predecessor's and unrelated to arrival waves. These args make the mapping EXPLICIT and
// host-chosen, which is the prerequisite for arrival-ordered consumption (spec appendix B, phase 1).
//
// In the FIXED prefix rather than the fused block because the on-chip ring runs on every path, including
// tp == 1 where there is no fused block at all.
enum RingSlotArg : uint32_t {
    kRsOwnSlot = 17,  // slot this core's OWN stripe occupies (i.e. the step at which it is consumed)
    // The slot this core's chunk occupies ON THE PEER DEVICE -- direct-L1's fabric destination. Equal to
    // kRsOwnSlot only while every device shares one schedule. Availability order does NOT: hop counts differ
    // per device, so the peer sorts its chunks differently and a relay that assumed its own slot writes into
    // the wrong one (measured: PCC 0.204). Two of them because a core can send both ways (a LINE origin, or a
    // ring origin under the balanced schedule) and the two neighbours have different schedules again.
    kRsPeerSlotFwd = 18,
    kRsPeerSlotBwd = 19,
    kRsFwdBase = 20,                     // (G-1) pairs {src_slot, dst_slot_on_successor}, one per forward step
    kRsArgCount = kRsFwdBase + 2u * 7u,  // G-1 == 7 forwards; == 34
};

// ---------------------------------------------------------------------------------------------------
// The ring schedule: ONE source of truth for the writer and the in1 reader.
// ---------------------------------------------------------------------------------------------------
// `consume_pos[p][s]` = which ring POSITION's stripe core at ring position `p` consumes at step `s`. Every
// other quantity either side needs is derived from it:
//
//   in1 reader : reads in1 block `consume_pos[p][s]*W + wb` at step s   (it must walk the identical global-K
//                order as the in0 side -- a spec requirement, and the desync this branch has hit twice)
//   writer     : own_slot   = the s where consume_pos[p][s] == p        (its own stripe)
//                src_slot   = s                                        (forward what it just consumed)
//                dst_slot   = the j where consume_pos[p+1][j] == consume_pos[p][s]
//                             i.e. the SUCCESSOR's slot for that same stripe -- inverting the successor's
//                             map is what makes the two sides provably consistent instead of coincidentally
//                             so, since both now come from this one array.
//
// Expressed in ring-position space, not core indices: the successor of position `p` is `p+1 mod G`, and
// consume order depends only on position, so no core-index bookkeeping is needed here.
//
// PHASE 2 fills it with today's rotation, `consume_pos[p][s] = (p - s) mod G`, which reproduces phase 1's
// hand-derived values exactly (asserted below). PHASE 3 replaces only this function.
struct RingSchedule {
    static constexpr uint32_t kG = 8u;
    uint32_t consume_pos[kG][kG]{};  // [ring position][step] -> ring position of the stripe consumed

    uint32_t slot_of(uint32_t p, uint32_t pos) const {
        for (uint32_t s = 0; s < kG; ++s) {
            if (consume_pos[p][s] == pos) {
                return s;
            }
        }
        TT_THROW("ring schedule: position {} never consumed by ring position {}", pos, p);
    }
};

// AVAILABILITY ORDER (spec appendix B phase 3). `fabric_hops[p]` is how many device-to-device hops the chunk
// fetched by ring position p has to make to reach this device; pass an empty span for the rotation.
//
// A chunk becomes usable at core c at `fabric_hops(o)*kWave + on_chip_hops(o->c)*kHop`, and the core should
// work on whatever is usable soonest. Today it instead consumes in ring order `c, c-1, c-2, ...`, which forces
// the two cores per ring whose own chunk is in the LAST fabric wave to idle through the entire gather and then
// owe their whole matmul -- that is the 19.6 us waiting term. See appendix B's worked example.
//
// Only the RATIO of the two constants matters, and only weakly: any ratio above G makes this "sort by fabric
// hops, break ties by on-chip distance", which is the intent. Measured, a fabric hop is ~12 us against ~0.45 us
// for an on-chip hop, so 100:1 is if anything conservative.
constexpr uint32_t kWaveCost = 100u;
constexpr uint32_t kHopCost = 1u;

inline RingSchedule build_ring_schedule(const uint32_t* fabric_hops) {
    RingSchedule sch;
    for (uint32_t p = 0; p < RingSchedule::kG; ++p) {
        for (uint32_t s = 0; s < RingSchedule::kG; ++s) {
            // Today's rotation: at step s, position p consumes the stripe that originated p-s positions back.
            sch.consume_pos[p][s] = (p + RingSchedule::kG - s) % RingSchedule::kG;
        }
    }
    if (fabric_hops != nullptr) {
        for (uint32_t c = 0; c < RingSchedule::kG; ++c) {
            // Stable insertion sort by availability; ties fall back to chunk index so the schedule is
            // deterministic (it must be: the host emits it and two kernels have to agree on it).
            uint32_t order[RingSchedule::kG];
            for (uint32_t o = 0; o < RingSchedule::kG; ++o) {
                order[o] = o;
            }
            auto avail = [&](uint32_t o) {
                return fabric_hops[o] * kWaveCost + ((c + RingSchedule::kG - o) % RingSchedule::kG) * kHopCost;
            };
            for (uint32_t i = 1; i < RingSchedule::kG; ++i) {
                uint32_t v = order[i], j = i;
                while (j > 0 && avail(order[j - 1]) > avail(v)) {
                    order[j] = order[j - 1];
                    --j;
                }
                order[j] = v;
            }
            for (uint32_t s = 0; s < RingSchedule::kG; ++s) {
                sch.consume_pos[c][s] = order[s];
            }
        }
    }
    // Every core must consume every stripe exactly once, or some global K tile is dropped or double-counted
    // on this device -- the failure the spec calls out first ("every global K tile is consumed once").
    for (uint32_t p = 0; p < RingSchedule::kG; ++p) {
        uint32_t seen = 0;
        for (uint32_t s = 0; s < RingSchedule::kG; ++s) {
            seen |= 1u << sch.consume_pos[p][s];
        }
        TT_FATAL(seen == 0xFFu, "ring schedule for position {} is not a permutation (mask {:#x})", p, seen);
    }
    return sch;
}

// Offsets WITHIN the fused-gather writer arg block, i.e. relative to fused_rt_base. The kernel reads this
// block sequentially, create_at pushes it in this order, and override_runtime_arguments patches the three
// entries that can change between invocations. All three places must agree, so they name these constants
// rather than counting; kFusedArgCount is checked against the actual push count at the end of the block.
enum FusedGatherArg : uint32_t {
    kFgIsClient = 0,
    kFgRank = 1,
    kFgTp = 2,
    kFgKShardTiles = 3,
    kFgStageAddr = 4,  // patched on replay (caller ping-pongs the staging buffer)
    kFgChunkReadySem = 5,
    kFgHasFwd = 6,
    kFgHasBwd = 7,
    kFgMtTotal = 8,
    kFgKtGlobal = 9,
    kFgShardAddr = 10,  // patched on replay (in0 may be a fresh allocation)
    kFgBankId = 11,     //
    kFgMyRecvSem = 12,  // THIS core's direction's credit; patched on replay (caller ping-pongs it)
    // Bidirectional schedule. One direction per core: masters 0..m_groups-1 drive forward, the rest
    // backward, and each direction's cores cover all of M between them. Only the mux for a core's own
    // direction is registered, so a core is never a client of a mux it does not drive.
    kFgDir = 13,  // 0 = forward, 1 = backward
    // Send and receive counts are SEPARATE because on a line they differ per rank: node d forwards the
    // d+1 shards that originated at 0..d but only ever receives d of them. On a ring the two coincide.
    kFgSendRounds = 14,
    kFgRecvRounds = 15,
    kFgMGroups = 16,  // M groups for the FABRIC split (= num_masters / 2); the local copy stays 8-way
    // On-chip gather barrier: every master reports to master 0, which multicasts one go-ahead to the grid.
    kFgIsMaster0 = 17,
    kFgGatherCountSem = 18,
    kFgNumMasters = 19,
    kFgMaster0X = 20,
    kFgMaster0Y = 21,
    // Progressive consumption: per-direction arrival publication.
    kFgDirCountSem = 22,
    kFgWaveFwdSem = 23,
    kFgWaveBwdSem = 24,
    kFgFwdCoordX = 25,
    kFgFwdCoordY = 26,
    kFgBwdCoordX = 27,
    kFgBwdCoordY = 28,
    kFgFwdRecvTotal = 29,  // arrivals the FORWARD stream will deliver (both needed by every core)
    kFgBwdRecvTotal = 30,
    kFgLocalDoneSem = 31,  // masters-finished-local-staging counter (one per master, not just master 0)
    // Blocked-cyclic K assignment (see the k-stripe comment at the runtime-arg push site). The writer needs
    // these to map its capacity-local K index onto a global staging column.
    kFgKRunLen = 32,
    kFgKStripeBase = 33,
    kFgKShardStride = 34,
    // Whether each coordinator's writer NOC traverses the grid opposite to NOC_0, i.e. whether it must
    // hand the multicast rectangle its corners swapped. Per-coordinator because the forward and backward
    // coordinators are different cores and need not share a NOC.
    kFgFwdCoordSwap = 35,
    kFgBwdCoordSwap = 36,
    // Capacity-local K slots reserved per source rank. Equals kFgKRunLen on the DRAM-staged path (so every
    // formula reduces to the old one); larger under direct-L1, which pads each rank up to a whole number of
    // ring slots so that a slot never straddles two ranks. See build_direct_l1_plan.
    kFgKRankSpan = 37,
    // ---- PHASE 2 direct-L1 stream plan for THIS core (all zero on the staged path). ----
    kFgDl1Active = 38,   // 1 => this core sources a real stripe (0 => its slot is pure zero padding)
    kFgDl1Dist = 39,     // hops from the origin; 0 => this device owns the rank and reads local DRAM
    kFgDl1SendFwd = 40,  // drives the forward mux
    kFgDl1SendBwd = 41,  // drives the backward mux (both, for a LINE origin)
    kFgDl1RecvSem = 42,  // GLOBAL semaphore address for this core's single arrival; patched on replay
    // Payload bytes per fabric packet. A RUNTIME arg rather than a compile-time one on purpose: every CT arg
    // added here displaces the TensorAccessorArgs indices below it, which builds cleanly and then fails all
    // 40 tests on PCC (this branch has done exactly that once).
    kFgDl1PacketBytes = 43,
    // 1 => this core may DEFER its own-chunk wait out of the prologue and into the ring step that consumes it,
    // so that while waiting it keeps relaying on the on-chip ring. Always safe for a leaf (nothing downstream
    // depends on it). For a RELAY it trades a later fabric hand-off to the next device against not freezing
    // the ring behind it -- which of those dominates is a measurement, hence TT_AGMM_DEFER_ALL.
    kFgDl1Defer = 44,
    kFgNumReleaseRanges = 45,
    // Followed by kFgNumReleaseRanges 6-word records {sx, sy, ex, ey, dests_fwd, dests_bwd} in VIRTUAL
    // coords: the multicast rectangles master 0 releases. Per-range rather than one bounding box because
    // the bank-adjacent placement is deliberately not a filled rectangle.
    kFusedArgCount = 46,  // + 6*num_release_ranges + 2*num_masters words, then the mux client block(s)
};

// Mux sizing. Kept deliberately small for bring-up: the design spec says to optimise for the default
// 4 KiB fabric packet, and one channel per direction is enough while a single ring does the forwarding.
// Channels are sized at run time to the number of clients that will ACTUALLY register with a given mux
// (masters-per-direction / num_links). Mux v2 self-terminates by counting close() calls against its
// compile-time channel count, so an over-provisioned mux simply never exits -- it does not error, it
// hangs. Every change to the client split has to move this number with it.
constexpr uint32_t kGatherScratchTiles = 8;  // must equal kGatherBatch in the writer kernel
constexpr uint8_t kMuxBuffersPerChannel = 8;
constexpr size_t kMuxChannelBufferBytes = 4096;  // 4 KiB packet

FusedGatherContext build_fused_gather_context(
    Program& program,
    const AllGatherRegimeAMatmulAsyncParams& attrs,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& in0,
    const std::vector<CoreCoord>& master_ring_cores,
    const CoreRangeSet& all_cores,
    IDevice* device,
    // Direct-L1: whether the path is active, and its client counts. These decide the mux channel counts, so
    // they have to be known BEFORE the muxes are created -- which is why the direct-L1 plan is built first.
    //
    // `dl1` is passed EXPLICITLY rather than inferred from the counts being non-zero. Inferring it breaks the
    // one case where direct-L1 is active with zero clients -- ABLATE_NOGATHER, which deletes the fabric on
    // purpose -- and the breakage is that the muxes get sized the STAGED way (num_links x channels_per_mux)
    // while nothing registers with them.
    bool dl1,
    uint32_t dl1_clients_fwd,
    uint32_t dl1_clients_bwd) {
    FusedGatherContext ctx;
    if (attrs.tp <= 1) {
        return ctx;  // single-chip path: no fabric, nothing to build
    }
    ctx.enabled = true;
    ctx.tp = attrs.tp;

    const auto topology = attrs.topology_is_ring ? ttnn::ccl::Topology::Ring : ttnn::ccl::Topology::Linear;
    ctx.rank = ttnn::ccl::get_linearized_index_from_physical_coord(in0, mesh_coordinate, attrs.cluster_axis);

    auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
    TT_FATAL(mesh_device != nullptr, "fused gather requires a MeshDevice");
    const auto src_node = mesh_device->get_fabric_node_id(mesh_coordinate);

    // A ring always has both neighbours; a line returns nullopt at its two ends, which is exactly the
    // "no such direction" gate the bidirectional schedule needs.
    const auto fwd_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, mesh_coordinate, 1, topology, attrs.cluster_axis);
    const auto bwd_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, mesh_coordinate, -1, topology, attrs.cluster_axis);

    // Readiness semaphore, on the matmul's own cores (receivers wait on it there).
    const CoreRangeSet ring_crs(std::vector<CoreRange>([&] {
        std::vector<CoreRange> v;
        v.reserve(master_ring_cores.size());
        for (const auto& c : master_ring_cores) {
            v.emplace_back(c, c);
        }
        return v;
    }()));
    // These two stay PROGRAM semaphores on purpose: both sides of the handshake are on this chip and in the
    // same program launch, so the cross-chip early-credit race that forced fwd_recv to be global does not
    // apply -- and dispatch re-zeroes them on every enqueue, which is exactly the re-arm we want.
    // Both live on ALL cores: every core waits on the go-ahead flag, not just the master ring.
    ctx.num_masters = static_cast<uint32_t>(master_ring_cores.size());
    // ---- The staged path's progressive-publication semaphores. NOT allocated under direct-L1. ----
    // Direct-L1 has no wave publication at all: a core gates on its own single arrival credit, so the
    // go-ahead flag, the local-staging barrier, the per-reporter slots and the two wave counters are all
    // dead there (the kernel (void)-casts their ids and never calls get_semaphore on them).
    //
    // Allocating them anyway is not free -- there is a HARD budget of 16 semaphores per core, and this is
    // 4 + num_masters/2 of them. Balanced bidirectional delivery makes each origin core a client of BOTH
    // muxes, and every client registration costs 2 more (the mux's flow-control pair), which tipped Sm=2
    // over the limit: "Cannot add semaphore on core 0-9. Max number of semaphores (16) reached!". Freeing the
    // dead ones is what pays for the second mux connection.
    if (!dl1) {
        ctx.chunk_ready_sem_id = CreateSemaphore(program, all_cores, 0);  // 0 == INVALID
        ctx.local_done_sem_id = CreateSemaphore(program, all_cores, 0);
    }
    // ONE SLOT PER REPORTER, not per arrival. Two constraints have to be met at once:
    //
    //  * A single shared counter is ambiguous -- with 4 masters, a cumulative count of 2*4 is produced
    //    equally by all four at arrival 2 and by two at arrival 4 with the other two empty, so it would
    //    publish a wave that has not landed. Silent wrong answer, not a hang.
    //  * One semaphore per ARRIVAL scales with tp and blew the 16-semaphore-per-core budget at tp=8
    //    ("Cannot add semaphore on core 0-9"), which fails program creation outright.
    //
    // Per-reporter slots satisfy both: master b only ever increments slot b, so slot b IS master b's
    // arrival count -- single writer, monotone, and a fixed m_groups slots regardless of tp. The
    // coordinator publishes arrival i once every slot has reached i.
    if (!dl1) {
        ctx.dir_count_sem_id = CreateSemaphore(program, all_cores, 0);
        const uint32_t reporter_slots = ctx.num_masters / 2u;
        for (uint32_t b = 1; b < reporter_slots; ++b) {
            const uint32_t nxt = CreateSemaphore(program, all_cores, 0);
            TT_FATAL(
                nxt == ctx.dir_count_sem_id + b,
                "per-reporter semaphores must be consecutive for base+b indexing, got {} after base {}",
                nxt,
                ctx.dir_count_sem_id);
        }
        ctx.wave_fwd_sem_id = CreateSemaphore(program, all_cores, 0);
        ctx.wave_bwd_sem_id = CreateSemaphore(program, all_cores, 0);
    }
    // Hard-tie the mux sizing to the client split. Getting these out of step does not fail loudly: the
    // mux's forwarder RISC just spins waiting for close() calls that never come.
    // Split each direction's masters across num_links muxes. Must divide evenly: an uneven split would
    // leave one mux short of the close() count it waits for, i.e. a hang with no diagnostic.
    ctx.num_links = attrs.num_links;
    const uint32_t masters_per_direction = ctx.num_masters / 2u;
    if (!dl1) {
        TT_FATAL(
            ctx.num_links >= 1u && masters_per_direction % ctx.num_links == 0u,
            "num_links={} must divide the {} masters that drive each direction; an uneven split leaves a mux "
            "waiting on a close() count that never arrives",
            ctx.num_links,
            masters_per_direction);
        ctx.channels_per_mux = masters_per_direction / ctx.num_links;
    }
    // Deal `total` clients round-robin over the links (client c -> link c % L), and give each mux exactly the
    // count that dealing produces. Fewer clients than links => fewer muxes, because a 0-channel mux is
    // rejected outright by FabricMuxV2Config.
    auto mux_channel_counts = [&](uint32_t total) {
        std::vector<uint8_t> counts;
        if (total == 0u) {
            return counts;
        }
        const uint32_t links = std::min<uint32_t>(ctx.num_links, total);
        for (uint32_t L = 0; L < links; ++L) {
            counts.push_back(static_cast<uint8_t>(total / links + ((L < total % links) ? 1u : 0u)));
        }
        return counts;
    };
    ctx.mux_channels_fwd = dl1 ? mux_channel_counts(dl1_clients_fwd)
                               : std::vector<uint8_t>(ctx.num_links, static_cast<uint8_t>(ctx.channels_per_mux));
    ctx.mux_channels_bwd = dl1 ? mux_channel_counts(dl1_clients_bwd)
                               : std::vector<uint8_t>(ctx.num_links, static_cast<uint8_t>(ctx.channels_per_mux));
    TT_FATAL(
        !attrs.gather_semaphores.empty(),
        "the fused gather (tp={}) needs at least one caller-supplied global semaphore for the cross-chip "
        "arrival credit; a program semaphore cannot be used here because a peer can credit it before this "
        "device's program has launched and zeroed it",
        attrs.tp);
    TT_FATAL(
        attrs.gather_semaphores.size() >= 2u,
        "the bidirectional fused gather needs TWO global semaphores (one credit counter per direction), "
        "got {}",
        attrs.gather_semaphores.size());
    ctx.fwd_recv_sem_addr = attrs.gather_semaphores[0].address();
    ctx.bwd_recv_sem_addr = attrs.gather_semaphores[1].address();

    const size_t mux_base_l1 = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // ---- Packet size: take it from the FABRIC, not from a literal ----
    // The mux slot holds header THEN payload (the v2 sender writes the payload at
    // slot + sizeof(PACKET_HEADER_TYPE)), so the usable payload is the channel buffer minus the header. The
    // fabric already publishes both numbers, and its channel buffer is exactly header + max_payload.
    //
    // kMuxChannelBufferBytes (4096) is BELOW the fabric's 4352-byte max payload, so hardcoding it capped
    // direct-L1 at ONE bf16 tile per packet -- half a packet's worth of payload per header, per mux slot
    // handoff, per credit. Using the fabric's own size fits two 2 KiB tiles instead. Staged path keeps the
    // 4096 literal: it is proven at that size and its per-master payloads are large anyway.
    const size_t fab_channel_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const size_t fab_max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    // Whole bf16 tiles only: slot 0 is a tile-granular buffer and a partial trailing tile would need its own
    // odd-sized transfer for no benefit.
    const uint32_t dl1_packet_bytes =
        dl1 ? static_cast<uint32_t>(std::max<size_t>(1u, fab_max_payload / plan::kTileBytesBf16) * plan::kTileBytesBf16)
            : plan::kTileBytesBf16;
    const size_t mux_channel_bytes = dl1 ? fab_channel_bytes : kMuxChannelBufferBytes;
    TT_FATAL(
        !dl1 || dl1_packet_bytes + tt::tt_fabric::get_tt_fabric_packet_header_size_bytes() <= mux_channel_bytes,
        "direct-L1 packet ({} B) plus header ({} B) does not fit the mux channel buffer ({} B)",
        dl1_packet_bytes,
        tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(),
        mux_channel_bytes);
    ctx.dl1_packet_bytes = dl1_packet_bytes;
    ctx.mux_channel_bytes = mux_channel_bytes;

    // ---- Mux channel DEPTH, sized to what actually fits ----
    // Direct-L1 turns every consumer core into a client, so one mux can carry tens of channels instead of the
    // staged path's 4, and its L1 map (dominated by channels * depth * packet) stops fitting: measured
    // 48 channels x 8 buffers x 4 KiB = 1.5 MB against a 1.5 MB worker L1. Shrink the buffer DEPTH rather than
    // the packet size -- the design spec says to optimise for the default 4 KiB packet, and depth is the term
    // that only costs pipelining. FabricMuxV2Config's own map check is the backstop if this estimate drifts.
    uint32_t mux_depth = kMuxBuffersPerChannel;
    // (published to ctx at the end of the sizing block below, for the observability log)
    {
        uint32_t max_ch = 0;
        for (const auto c : ctx.mux_channels_fwd) {
            max_ch = std::max<uint32_t>(max_ch, c);
        }
        for (const auto c : ctx.mux_channels_bwd) {
            max_ch = std::max<uint32_t>(max_ch, c);
        }
        // One stream register per channel, 64 per Tensix worker -- a hard hardware cap that no depth can buy
        // back. num_links is the only lever, since a mux binds exactly one link.
        TT_FATAL(
            max_ch <= 64u,
            "TT_AGMM_DIRECT_L1 needs {} channels on one mux (fwd={} bwd={} clients over num_links={}), over "
            "the 64 stream registers a Tensix worker has. Raise num_links (a mux binds one link, so links are "
            "what add mux cores) or unset TT_AGMM_DIRECT_L1 to use the DRAM-staged path.",
            max_ch,
            dl1_clients_fwd,
            dl1_clients_bwd,
            ctx.num_links);
        if (max_ch > 0u) {
            // Per-channel L1 beyond the payload buffers: connection info + handshake + credit scratch, each
            // L1-aligned. Rounded generously; the shared regions (status, trid ring, control block) are the
            // small constant.
            constexpr size_t kPerChannelOverheadBytes = 512;
            constexpr size_t kSharedOverheadBytes = 16u * 1024u;
            const size_t budget = device->l1_size_per_core() - mux_base_l1;
            while (mux_depth > 1u &&
                   max_ch * (static_cast<size_t>(mux_depth) * mux_channel_bytes + kPerChannelOverheadBytes) +
                           kSharedOverheadBytes >
                       budget) {
                mux_depth /= 2u;
            }
        }
    }
    ctx.mux_depth = mux_depth;

    // The mux cores must not collide with the matmul's compute cores. The matmul occupies the low part of
    // the grid (banks x ring groups), so take mux cores from the TOP row downward.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto mux_core_for = [&](uint32_t slot) { return CoreCoord{grid.x - 1u - slot, grid.y - 1u}; };

    auto deploy = [&](const std::optional<ttnn::MeshCoordinate>& dst_coord,
                      uint32_t slot_base,
                      const std::vector<uint8_t>& channels,
                      std::vector<std::unique_ptr<tt::tt_fabric::FabricMuxV2Config>>& cfgs_out,
                      std::vector<CoreCoord>& vcores_out,
                      std::vector<uint8_t>& next_channel_out) {
        if (!dst_coord.has_value()) {
            return;  // line end: this direction does not exist
        }
        for (uint32_t L = 0; L < channels.size(); ++L) {
            const CoreCoord mux_logical = mux_core_for(slot_base + L);
            cfgs_out.push_back(std::make_unique<tt::tt_fabric::FabricMuxV2Config>(
                channels[L], static_cast<uint8_t>(mux_depth), mux_channel_bytes, mux_base_l1));
            tt::tt_fabric::add_fabric_mux_v2_to_program(
                program,
                *cfgs_out.back(),
                mux_logical,
                src_node,
                mesh_device->get_fabric_node_id(*dst_coord),
                /*link_idx=*/L,
                tt::tt_metal::NOC::RISCV_0_default);
            vcores_out.push_back(device->worker_core_from_logical_core(mux_logical));
            next_channel_out.push_back(0);
        }
    };

    // Mux cores are taken from the top row: forward links occupy slots [0, num_links), backward the next
    // num_links, so the two directions never share a core.
    // mux_core_for walks the top row leftward from grid.x-1, so the two directions together must fit in it.
    // The design doc flags this as the one bounds check widening the client set past 8 masters needs.
    TT_FATAL(
        ctx.mux_channels_fwd.size() + ctx.mux_channels_bwd.size() <= grid.x,
        "the fused gather needs {} mux cores (fwd) + {} (bwd) from the {}-wide top row",
        ctx.mux_channels_fwd.size(),
        ctx.mux_channels_bwd.size(),
        grid.x);
    deploy(fwd_coord, 0, ctx.mux_channels_fwd, ctx.mux_cfgs_fwd, ctx.mux_virtual_cores_fwd, ctx.next_channel_fwd);
    // Both muxes are deployed now that the kernel drives both directions. Mux v2 self-terminates by
    // counting close() calls against its compile-time channel count and has no host-side termination
    // signal, so a mux must only ever be given clients that really do open/close it -- which is why each
    // master registers with the mux for its OWN direction only.
    deploy(
        bwd_coord,
        static_cast<uint32_t>(ctx.mux_channels_fwd.size()),
        ctx.mux_channels_bwd,
        ctx.mux_cfgs_bwd,
        ctx.mux_virtual_cores_bwd,
        ctx.next_channel_bwd);
    return ctx;
}

}  // namespace

ttnn::device_operation::CachedProgram<AllGatherRegimeAMatmulAsyncProgramFactory::shared_variables_t>
AllGatherRegimeAMatmulAsyncProgramFactory::create_at(
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Program program = CreateProgram();

    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    Tensor& out = tensor_return_value[0];  // chunk 0 (or the sole output when chunks==1)
    IDevice* device = in0.device();

    // get_worker_noc_hop_distance() hard-asserts a unit MeshDevice, so resolve a single representative
    // device for the placement queries below. Those queries only drive core PLACEMENT heuristics (M-split
    // worker siting and in0 ring order) — never the math — so a multi-device mesh can use device 0's
    // physical layout without affecting correctness.
    //
    // CAVEAT: on this Galaxy every chip harvests a DIFFERENT Tensix column, so logical->physical layouts
    // are not identical across the mesh and the placement chosen from device 0 may be mildly suboptimal
    // elsewhere. Acceptable for Phase 0 (correctness first); revisit when tuning fused performance.
    IDevice* noc_ref_device = device;
    if (auto* mesh = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
        mesh != nullptr && mesh->num_devices() > 1) {
        noc_ref_device = mesh->get_devices().front();
    }

    // Resolve config=None via the auto-selector (deterministic in the tile dims, program-cache-safe).
    // K for PLANNING is always the GLOBAL K, taken from in1 (which is full-K on every device). With the
    // fused gather in0 only carries this rank's [M, K/tp] shard, so planning off in0 would size the whole
    // matmul to a fraction of the real K. in1 is the authority; validate has already checked
    // in1.K == in0.K * tp.
    const uint32_t Mt_r = (static_cast<uint32_t>(in0.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(in1.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = operation_attributes.config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));

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
    // Plan against the GATHERED activation, not this device's shard. make_and_build_plan takes Mt/Kt from
    // in0's logical shape, so handing it the shard plans the whole matmul for K/tp: geo.Kt (and with it the
    // in0 row stride, the k-slice capacity, and the fused block's k_shard_tiles) all come out tp times too
    // small, and every page index the gather computes is wrong. The staging buffer already has exactly the
    // [M, K_global] shape the matmul will actually read, so it is the correct planning input.
    const Tensor& in0_for_plan = (operation_attributes.tp > 1) ? *tensor_args.gather_staging_buffer : in0;
    auto planres = make_and_build_plan(device, in0_for_plan, in1, cfg, fusion);
    TT_FATAL(planres.ok(), "all_gather_regime_a_matmul_async planner rejected config: {}", planres.error);
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
    // ---- Timing ablations (TT_AGMM_ABLATE). These produce WRONG RESULTS on purpose; timing only. ----
    //   nowait   : the gather runs in full and publishes, but consumers never gate on arrival. The delta
    //              against the real number is the pure DEPENDENCY STALL.
    //   nogather : additionally, gather cores stage locally but send nothing. Isolates the matmul + local
    //              staging floor inside the fused program.
    //
    // Measured on medium/tp4/ring/2-link, us per invocation:
    //   matmul alone ~83   |   nowait 113.2   |   full fused 150.2   |   Phase-0 125.3
    // i.e. ~30 us is the gather competing for DRAM/NoC bandwidth even when nothing waits on it (overlap
    // cannot recover that, only a cheaper gather can), and ~37 us is pure stall (recoverable -- removing
    // it alone would land below Phase-0). Host dispatch time is not evidence either way; this is device
    // FW duration from the tracy device profiler, per the design spec's requirement.
    const char* ablate_env = std::getenv("TT_AGMM_ABLATE");
    const std::string ablate = ablate_env ? ablate_env : "";
    std::map<std::string, std::string> wdefs;
    // COMPOSABLE: TT_AGMM_ABLATE is matched by substring, so `nopayload,nowait` turns on both. It used to be
    // an if/else chain, which silently made the two mutually exclusive -- and the one cell that needs BOTH is
    // the one that separates the fabric's fixed cost from its latency cost:
    //
    //                     payload            no payload
    //     wait            full               nopayload
    //     no wait         nowait             nopayload,nowait   <-- unreachable before this
    //
    //   nopayload         - floor            = connections + credits + waiting on them, zero bytes
    //   nopayload,nowait  - floor            = connections + credits alone
    //   difference of the two                = what waiting on a credit costs when data is instant
    //
    // `nopayload` drops the fabric PAYLOAD writes and nothing else: mux connections still open and close,
    // credits still cross the fabric, and consumers still wait on them, so the protocol, the client count,
    // the hop structure and the arrival dependency are all held fixed while only the bytes go away.
    // `nogather` cannot express any of this -- it deletes the muxes and the dependency structure too.
    const bool ab_nogather = ablate.find("nogather") != std::string::npos;
    const bool ab_nopayload = ablate.find("nopayload") != std::string::npos;
    // nogather implies nowait: with nothing sent, a consumer that still waited would hang.
    const bool ab_nowait = ab_nogather || ablate.find("nowait") != std::string::npos;
    if (ab_nogather) {
        wdefs["ABLATE_NOGATHER"] = "1";
    }
    if (ab_nopayload) {
        wdefs["ABLATE_NOPAYLOAD"] = "1";
    }
    if (ab_nowait) {
        wdefs["ABLATE_NOWAIT"] = "1";
    }
    TT_FATAL(
        ablate.empty() || ab_nogather || ab_nopayload || ab_nowait,
        "TT_AGMM_ABLATE='{}' matched no known ablation (nogather, nopayload, nowait; comma-separate to "
        "combine). Refusing rather than silently measuring the unablated program as if it were ablated.",
        ablate);
    // Fused fabric all-gather. A PREPROCESSOR define, not just a compile-time arg: the prologue declares a
    // TensorAccessorArgs that only exists when tp > 1, and `if constexpr` does NOT discard an ill-formed
    // branch in a non-template function -- it would still be compiled and fail deduction on the tp == 1 build.
    if (operation_attributes.tp > 1) {
        wdefs["FUSED_GATHER"] = "1";
    }
    // ---- PHASE 2: direct-L1 streaming (TT_AGMM_DIRECT_L1=1). Opt-in; DRAM staging stays the default and
    // the A/B oracle, per the design spec's "retain DRAM staging as an A/B diagnostic until direct L1 is
    // proven correct and faster". See tools/mm_sweep/AGMM_DIRECT_L1_DESIGN.md.
    //
    // WHY: this shape is DRAM-bandwidth-bound (83.0 us == 30.14 MB / 363 GB/s == 81% of the 448 GB/s
    // Galaxy RevB peak), so surplus bytes convert directly into time. Phase 1's staging round-trip costs
    // 5.25 MB/device, putting its ROOFLINE at 97.5 us -- above the 91.3 us gate (1.1 * max(83.0, 41.3)).
    // Phase 1 therefore cannot pass however well it is scheduled. Direct-L1 keeps the gathered activation
    // out of DRAM entirely: 28.2 MB -> 77.6 us, below even the single-chip matmul.
    //
    // No credits and no bounded window: cb0 is already sized for the worker's COMPLETE gathered K/Pk slice
    // (plan.hpp compute_cb_sizes, planned against the staging buffer -- see in0_for_plan above), so a
    // remote stripe written into its final cb0 slot is never overwritten. Slot 0 is written only by fabric
    // (remote positions) or DRAM (the local position); slot s>0 only by the ring forward. Zero reuse
    // anywhere, so the only synchronisation needed is the per-shard arrival semaphores that already exist.
    const bool direct_l1 = (std::getenv("TT_AGMM_DIRECT_L1") != nullptr) && operation_attributes.tp > 1;
    if (direct_l1) {
        // Ns>1 ring groups need IDENTICAL in0, so a per-consumer fabric scatter would emit duplicate copies
        // of the same tiles across the fabric -- which the design spec forbids ("Ns groups need identical A;
        // they may share forwarding work, but must not emit duplicate fabric copies"). Doing it properly
        // needs a NoC replication step after ingress. Until then refuse rather than silently paying (Ns)x
        // the fabric bytes; DRAM staging still covers Ns>1.
        TT_FATAL(
            cfg.n_slices <= 1u,
            "TT_AGMM_DIRECT_L1 does not support Ns>1 yet (got Ns={}): Ns groups consume identical in0, so a "
            "direct per-consumer scatter would send each tile over the fabric Ns times. Unset "
            "TT_AGMM_DIRECT_L1 to use the DRAM-staged path, which handles Ns>1.",
            cfg.n_slices);
        wdefs["DIRECT_L1"] = "1";
    }
    // extra COMPUTE defines beyond fusion; currently only the reduction strategy (RSCATTER).
    std::map<std::string, std::string> cdefs_extra;

    // ---- INTERNAL REDUCTION STRATEGY: linear chain (default) vs ring REDUCE-SCATTER. ----
    // The chain sends each of the Pk-1 non-root bands' FULL output block one hop up, so the last partial only
    // starts moving after Pk-2 earlier hops and the root alone writes all the output. Reduce-scatter instead
    // tile-partitions the block into Pk chunks and rotates them around the Pk cores: the same total number of
    // adds and the same total bytes, but every core sends concurrently every round, and each core ends up
    // owning + writing ONE fully-reduced chunk, so the output write is spread over Pk cores instead of 1.
    //
    // Adopted only on the regime where it was MEASURED to win 5-9% with zero regressions (five corpus shapes:
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
    // OBSERVABILITY ONLY (TT_REGIME_A_LOG_CFG): report what the picker and the internal gates actually chose.
    // Runs once per program-cache miss and changes NOTHING about behaviour -- there is no way to read the
    // auto-selected config from Python otherwise, and reporting a host-side mirror of the picker risks silently
    // misreporting if the mirror drifts from auto_select_config.
    if (std::getenv("TT_REGIME_A_LOG_CFG") != nullptr) {
        log_info(
            tt::LogOp,
            "regime_a_cfg M={} K={} N={} pick=({},{},{},{},{}) cores={} reduction={} placement={}",
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
        place_m_split_workers(P, noc_ref_device, geo);
    }

    // ---- Physical-topology-aware in0 ring ordering (PARETO) over each (kk,nn) group's Sm mm-rings. ----
    optimize_in0_ring_order(P, noc_ref_device, geo, Sm);

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

    // ---- PHASE 1 fused fabric all-gather: per-device host context ----
    // The MASTER in0 ring is ring group p == 0, i.e. core indices i = b*preaders for b in 0..7. Those 8
    // cores are the only fabric clients for now (load-balancing forwarding across the other rings is a
    // deliberate follow-up). Built here, after `cores` exists and before runtime args are emitted.
    const uint32_t preaders_pf = geo.num_cores / 8u;
    std::vector<CoreCoord> master_ring_cores;
    master_ring_cores.reserve(8);
    for (uint32_t b = 0; b < 8u; ++b) {
        master_ring_cores.push_back(cores[b * preaders_pf]);
    }

    // ---- PHASE 2 direct-L1: per-core stream plan, built BEFORE the mux context. ----
    // The mux channel count must equal the number of clients that will really open/close it, and under
    // direct-L1 that count is a property of this plan (on a LINE it even varies with this device's rank), so
    // the plan has to exist before the muxes are created. ring_pos is final by now: optimize_in0_ring_order
    // and the M-split placement both ran above.
    std::vector<DirectL1Core> dl1_plan, dl1_plan_fwd, dl1_plan_bwd;
    uint32_t dl1_clients_fwd = 0, dl1_clients_bwd = 0;
    if (direct_l1) {
        dl1_plan = build_direct_l1_plan(
            P,
            geo,
            Pk,
            kb,
            operation_attributes.tp,
            ttnn::ccl::get_linearized_index_from_physical_coord(
                in0, mesh_coordinate, operation_attributes.cluster_axis),
            operation_attributes.topology_is_ring,
            // BALANCED BIDIRECTIONAL delivery (spec appendix A), opt-in via TT_AGMM_DIRECT_L1_BALANCED=1.
            //
            // It halves the hop depth (tp=4: 3->2, tp=8: 7->4) and it is what the spec asks for, but it is NOT
            // the default because it measures WORSE on the dependent path. medium/ring/2 links, device
            // makespan, 48 timed iterations per sample:
            //
            //                    full                                     nowait          stall
            //   tp=4 parity      120.19                                   101.26          18.9
            //   tp=4 balanced    120.16  (=, within noise)                 99.89 (-1.4)   20.3
            //   tp=8 parity      136.4   median of 136.1/140.6/136.6/134.8 104.34          31.8
            //   tp=8 balanced    141.6   median of 141.0/138.4/141.6/142.1/141.9
            //                                                             101.83 (-2.5)   39.2
            //
            // tp=8 needed REPEATED INTERLEAVED sampling to call: between-process spread reaches 5 us, and a
            // single confirmation pair came out reversed (default 140.6 vs balanced 138.4) before four more
            // samples separated them cleanly. Do not re-decide this from one run each.
            //
            // Consistent in both directions: balanced makes the FABRIC cheaper (nowait improves at both tp)
            // and the STALL worse. Depth buys latency, but total link bytes are unchanged -- every stripe
            // still crosses tp-1 hops either way -- so on a shape that is NoC-occupancy-bound there is no
            // throughput to win, while the arrival pattern gets burstier: balanced delivers 2K/tp per wave
            // from both neighbours at once instead of K/tp, concentrating ingress against the on-chip ring.
            // That is the same mechanism that made the deferred fabric drain worse.
            //
            // Kept because it is correct (32/40, identical to the default) and is expected to win once the
            // dependency stall is addressed by per-wave rings, at which point the burstiness stops gating.
            // Re-measure it then before adopting.
            std::getenv("TT_AGMM_DIRECT_L1_BALANCED") != nullptr);
        // ABLATE_NOGATHER: strip the fabric out of the plan rather than out of the kernel. Clearing the send
        // flags here means the client counts below come out 0, so NO mux is deployed, no client block is
        // appended to the writer args, and the kernel's existing `send_fwd || send_bwd` guard skips the
        // senders -- no ablation #ifdef in the kernel at all.
        //
        // Removing the sends from the KERNEL while leaving the muxes deployed is what hung: mux v2
        // self-terminates by counting close() calls against its compile-time channel count, so a client that
        // never opens or closes leaves the forwarder RISC spinning. Deleting the mux is the only way to
        // ablate its traffic. `nogather` implies `nowait` (set together above), so the cores that would have
        // received a stripe do not wait for one.
        //
        // What this leaves running is the intended floor: the origin cores' local shard reads, the on-chip
        // ring, and the matmul -- everything except cross-device traffic.
        if (ab_nogather) {  // substring-matched above, so it composes with the other ablations
            for (auto& d : dl1_plan) {
                d.send_fwd = false;
                d.send_bwd = false;
            }
        }
        for (const auto& d : dl1_plan) {
            dl1_clients_fwd += d.send_fwd ? 1u : 0u;
            dl1_clients_bwd += d.send_bwd ? 1u : 0u;
        }
        // The two NEIGHBOUR devices' plans, built with the identical function so their hop counts -- and hence
        // their availability order -- cannot drift from how they compute it themselves. Only their hop counts
        // are used, to work out which slot each neighbour puts our chunk in.
        const uint32_t rank_self = ttnn::ccl::get_linearized_index_from_physical_coord(
            in0, mesh_coordinate, operation_attributes.cluster_axis);
        for (uint32_t which = 0; which < 2u; ++which) {
            const uint32_t nb_rank = which == 0u ? (rank_self + 1u) % operation_attributes.tp
                                                 : (rank_self + operation_attributes.tp - 1u) % operation_attributes.tp;
            (which == 0u ? dl1_plan_fwd : dl1_plan_bwd) = build_direct_l1_plan(
                P,
                geo,
                Pk,
                kb,
                operation_attributes.tp,
                nb_rank,
                operation_attributes.topology_is_ring,
                std::getenv("TT_AGMM_DIRECT_L1_BALANCED") != nullptr);
        }
    }

    FusedGatherContext fused_gather = build_fused_gather_context(
        program,
        operation_attributes,
        mesh_coordinate,
        in0,
        master_ring_cores,
        all_cores,
        device,
        direct_l1,
        dl1_clients_fwd,
        dl1_clients_bwd);

    // OBSERVABILITY ONLY: the direct-L1 fabric shape. Reported because every one of these numbers is derived
    // rather than configured -- packet size from the fabric's max payload, client counts from the stream plan,
    // channel depth from what fits L1 -- so "it silently did something else" is otherwise indistinguishable
    // from "this knob does not matter". Packet size in particular: measured at 2048 vs 4096 the makespan moved
    // 0.5 us, and that conclusion is only meaningful if 4096 was really in use.
    if (direct_l1 && std::getenv("TT_REGIME_A_LOG_CFG") != nullptr) {
        log_info(
            tt::LogOp,
            "regime_a_direct_l1 packet={}B clients=({},{}) muxes=({},{}) max_ch={} depth={} channel={}B",
            fused_gather.dl1_packet_bytes,
            dl1_clients_fwd,
            dl1_clients_bwd,
            fused_gather.mux_channels_fwd.size(),
            fused_gather.mux_channels_bwd.size(),
            [&] {
                uint32_t m = 0;
                for (const auto c : fused_gather.mux_channels_fwd) {
                    m = std::max<uint32_t>(m, c);
                }
                for (const auto c : fused_gather.mux_channels_bwd) {
                    m = std::max<uint32_t>(m, c);
                }
                return m;
            }(),
            fused_gather.mux_depth,
            fused_gather.mux_channel_bytes);
    }

    // ---- On-chip gather barrier geometry ----
    // A master core's fwd_recv count only proves ITS OWN M slice arrived: the gather splits M by bank_id,
    // while the matmul splits M by the planner's m_start. Those partitions differ, so every core -- master
    // or not -- can read rows that a DIFFERENT core gathered. Each master therefore reports completion to
    // master 0, which multicasts a single go-ahead to the whole grid.
    if (fused_gather.enabled) {
        // worker_core_from_logical_core returns TRANSLATED (virtual) coords, in which the Blackhole grid is
        // dense whatever the harvesting mask. Raw physical coords would straddle harvested columns here.
        fused_gather.master0_virtual = device->worker_core_from_logical_core(master_ring_cores[0]);
        for (const auto& mc : master_ring_cores) {
            fused_gather.master_virtuals.push_back(device->worker_core_from_logical_core(mc));
        }
        const uint32_t mg = fused_gather.num_masters / 2u;
        fused_gather.fwd_coord_virtual = device->worker_core_from_logical_core(master_ring_cores[0]);
        fused_gather.bwd_coord_virtual = device->worker_core_from_logical_core(master_ring_cores[mg]);

        // Release the grid with ONE MULTICAST PER CoreRange. A single bounding-box mcast is not an option:
        // the bank-adjacent placement is deliberately not a filled rectangle, so the box also covers cores
        // running no kernel of ours. Per-range keeps the arg cost at 5 words per rectangle instead of 2 per
        // core, which is what makes the 80- and 96-core shapes fit.
        //
        // Non-loopback multicast throughout: master 0 sets its own flag directly, and this variant
        // explicitly does not write to self, so any range containing master 0 must not count it in
        // num_dests.
        // Which NOC master 0's WRITER uses -- note the inversion: NoC group 0 runs writerA, which is
        // created on RISCV_1 / NOC::RISCV_1_default, i.e. NOC_1. Group 1 runs writerB on NOC_0.
        // A multicast rectangle must be given its corners in the issuing NOC's traversal order, and NOC_1
        // traverses opposite to NOC_0, so group 0 is exactly the case that needs the swap.
        //
        // Getting this backwards is near-invisible on a single ring: those 8 cores are bank-adjacent and
        // scattered, so merge_ranges leaves them as 1x1 rectangles where start == end and the swap is a
        // no-op. It only bites once several cores actually merge into a rectangle.
        // Rectangles are stored in NOC_0 corner order. The swap is applied per-SENDER in the kernel,
        // because the forward and backward coordinators are different cores: deciding it here from
        // master 0 alone was correct only by accident (every master is slice p == 0, so they all land on
        // the same NOC). NoC group 0 runs writerA on RISCV_1 / NOC_1, hence the inversion.
        const uint32_t mgc0 = fused_gather.num_masters / 2u;
        fused_gather.fwd_coord_swaps = (core_noc[0] == 0u) ? 1u : 0u;
        fused_gather.bwd_coord_swaps = (core_noc[mgc0 * preaders_pf] == 0u) ? 1u : 0u;
        // all_cores was built one CoreRange(c, c) per core and CoreRangeSet does NOT auto-coalesce, so
        // without this the "per-range" multicast degenerates to one rectangle per core (79 for an 80-core
        // grid) and buys nothing.
        const CoreRangeSet release_crs = all_cores.merge_ranges();
        for (const auto& cr : release_crs.ranges()) {
            const CoreCoord s = device->worker_core_from_logical_core(cr.start_coord);
            const CoreCoord e = device->worker_core_from_logical_core(cr.end_coord);
            const uint32_t mgc = fused_gather.num_masters / 2u;
            const uint32_t dests_fwd = static_cast<uint32_t>(cr.size()) - (cr.contains(master_ring_cores[0]) ? 1u : 0u);
            const uint32_t dests_bwd =
                static_cast<uint32_t>(cr.size()) - (cr.contains(master_ring_cores[mgc]) ? 1u : 0u);
            if (dests_fwd == 0u && dests_bwd == 0u) {
                continue;  // a range holding only a coordinator: it sets its own copy directly
            }
            fused_gather.release_ranges.push_back(
                {static_cast<uint32_t>(s.x),
                 static_cast<uint32_t>(s.y),
                 static_cast<uint32_t>(e.x),
                 static_cast<uint32_t>(e.y),
                 dests_fwd,
                 dests_bwd});
        }
        TT_FATAL(
            fused_gather.release_ranges.size() <= 48u,
            "fused gather barrier needs {} multicast rectangles to cover the worker set, over the 48 the "
            "runtime-arg budget allows",
            fused_gather.release_ranges.size());
    }

    // NOTE: packet headers come from PacketHeaderPool (a per-RISC L1 region), NOT from a circular buffer.
    // The CB carve-out is the older pattern; the pool is what current fabric kernels use.

    // ---- Circular buffers (spec §5) on all cores ----
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in0 k-slice resident
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // in1 (depth 4)
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);  // out
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);    // fp32 intermediate
    if (fused_gather.enabled) {
        // DEDICATED gather scratch. It used to be cb0's head via get_write_ptr(in0_cb) -- but
        // cb_reserve_back does not move the write pointer, so that address IS ring slot 0, and slots
        // 1..k sit immediately after it. While the coarse barrier existed nothing could fill those slots
        // during the gather. Under progressive consumption a non-master core (which has no fabric work)
        // clears its gate immediately and writes its shard into a master's slot 1 while that master is
        // still DMA-ing through its scratch -- the scratch overwrites the delivered shard, the master
        // then sees the ring credit and consumes garbage.
        //
        // It only bites when W*M_block*K_block < kGatherBatch, i.e. when a ring shard is smaller than the
        // batch: the small shape at Pk>=2. Staging stays byte-exact throughout, which is exactly why the
        // readback diagnostic looked clean.
        mkcb(program, all_cores, 11, kGatherScratchTiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
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
    // Index in the writer's runtime args where the fused-gather block begins. MUST mirror the wa push
    // order below exactly: 17 fixed, then bias, ternary, chunk, and reduce-scatter args. Asserted against
    // the real wa.size() at emission time so the two can never silently drift apart.
    const uint32_t fused_rt_base = kRsArgCount + (has_bias ? 1u : 0u) + (has_ternary ? 3u : 0u) +
                                   ((n_chunks > 1u) ? (2u + (n_chunks - 1u)) : 0u) + (rscatter ? 7u : 0u);

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
        use_reduce,            // 13
        // ---- fused fabric all-gather (Phase 1) ----
        // 14: 0 => the whole gather prologue compiles out, so tp==1 is byte-identical.
        // 15: index into the writer's runtime args where the fused block starts. Passed explicitly because
        //     the block sits after a variable number of optional bias / ternary / chunk / rscatter args.
        (operation_attributes.tp > 1) ? 1u : 0u,  // 14 fused_gather_enabled
        fused_rt_base};                           // 15 fused_rt_base
    // in0 ACCESSOR: on the fused path the matmul reads the GATHERED activation out of the staging buffer,
    // not the local shard. The staging buffer is [M, K_global] so the existing (m*Kt + k) addressing and the
    // geo.in0_stride_k row stride are already correct for it -- the matmul body needs no change at all, the
    // accessor just points somewhere else.
    const Tensor& in0_for_matmul = (operation_attributes.tp > 1) ? *tensor_args.gather_staging_buffer : in0;
    TensorAccessorArgs(*in0_for_matmul.buffer()).append_to(wct);
    TensorAccessorArgs(*out.buffer()).append_to(wct);
    // Fused-operand accessors, in the order the writer kernel expects: bias, then residual/gate.
    if (has_bias) {
        TensorAccessorArgs(*tensor_args.bias_tensor->buffer()).append_to(wct);
    }
    if (has_ternary) {
        TensorAccessorArgs(*tensor_args.fused_ternary_input_a->buffer()).append_to(wct);
        TensorAccessorArgs(*tensor_args.fused_ternary_input_b->buffer()).append_to(wct);
    }
    // LOCAL-SHARD accessor for the fused gather. Appended LAST, after every existing accessor, so adding it
    // cannot shift the indices the kernel already resolves by chaining. Only present/read when tp > 1.
    if (operation_attributes.tp > 1) {
        TensorAccessorArgs(*in0.buffer()).append_to(wct);
    }

    // Split-NOC: reader on the core's in1 NoC, writer on the OTHER NoC.
    //   g0 (noc==0): reader RISCV_0/NOC0, writer RISCV_1/NOC1
    //   g1 (noc==1): reader RISCV_1/NOC1, writer RISCV_0/NOC0
    // in1 reader takes no compile defines.
    std::map<std::string, std::string> rdefs;
    if (fused_gather.enabled) {
        // The in1 reader has to walk the SAME global-K order as the in0 side; the spec is explicit about
        // that. Give it the define so it parses the K-stripe args below.
        rdefs["FUSED_GATHER"] = "1";
    }
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
    TT_FATAL(
        sbh * sbw <= 4u, "all_gather_regime_a_matmul_async subblock {}x{} exceeds the 4-tile fp32 DST limit", sbh, sbw);
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
    const uint32_t in0_addr = in0_for_matmul.buffer()->address();  // staging buffer when tp>1
    const uint32_t in1_addr = in1.buffer()->address();
    const uint32_t out_addr = out.buffer()->address();

    // ONE ring schedule PER SLICE: both the writer's slot args and the in1 reader's consume order are derived
    // from it below, so the two cannot disagree about the global-K order.
    //
    // Per slice rather than one global schedule because availability ordering depends on how many fabric hops
    // each chunk needed, which is a property of the slice's own stream plan. The rotation (used on the staged
    // path, and when TT_AGMM_RING_ROTATION=1 forces it for A/B) is slice-independent.
    const bool ring_rotation = std::getenv("TT_AGMM_RING_ROTATION") != nullptr;
    std::vector<RingSchedule> ring_sch(preaders_pf), ring_sch_fwd, ring_sch_bwd;
    {
        // hops[slice][position] -- the fabric hop count of the chunk that ring position fetches.
        std::vector<std::array<uint32_t, RingSchedule::kG>> hops(preaders_pf);
        if (direct_l1 && !ring_rotation) {
            for (uint32_t i = 0; i < geo.num_cores; ++i) {
                hops[P.cores[i].slice][P.cores[i].ring_pos] = dl1_plan[i].dist;
            }
        }
        for (uint32_t sl = 0; sl < preaders_pf; ++sl) {
            ring_sch[sl] = build_ring_schedule((direct_l1 && !ring_rotation) ? hops[sl].data() : nullptr);
        }
        // Same construction for the two neighbours, so we can ask where THEY put our chunk.
        if (direct_l1 && !ring_rotation) {
            std::vector<std::array<uint32_t, RingSchedule::kG>> hf(preaders_pf), hb(preaders_pf);
            for (uint32_t i = 0; i < geo.num_cores; ++i) {
                hf[P.cores[i].slice][P.cores[i].ring_pos] = dl1_plan_fwd[i].dist;
                hb[P.cores[i].slice][P.cores[i].ring_pos] = dl1_plan_bwd[i].dist;
            }
            ring_sch_fwd.resize(preaders_pf);
            ring_sch_bwd.resize(preaders_pf);
            for (uint32_t sl = 0; sl < preaders_pf; ++sl) {
                ring_sch_fwd[sl] = build_ring_schedule(hf[sl].data());
                ring_sch_bwd[sl] = build_ring_schedule(hb[sl].data());
            }
        }
        if (direct_l1 && !ring_rotation && std::getenv("TT_REGIME_A_LOG_CFG") != nullptr) {
            std::string s;
            for (uint32_t p = 0; p < RingSchedule::kG; ++p) {
                s += fmt::format(" p{}(h{}):", p, hops[0][p]);
                for (uint32_t k = 0; k < RingSchedule::kG; ++k) {
                    s += fmt::format("{}{}", k ? "," : "", ring_sch[0].consume_pos[p][k]);
                }
            }
            log_info(tt::LogOp, "regime_a_ring_order slice0{}", s);
        }
    }

    auto phys = [&](uint32_t core_idx) {
        const auto& c = P.cores[core_idx].coord;
        return device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
    };

    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const plan::CorePlan& cp = P.cores[i];
        const KernelHandle rh = cp.noc ? readerB : readerA;
        const KernelHandle wh = cp.noc ? writerB : writerA;

        // ---- BLOCKED-CYCLIC global-K assignment for the fused path ----
        // The planner gives each Pk group a CONTIGUOUS K range. On the fused path that is the failure mode
        // the design spec calls out by name: gathered K is laid out contiguously by source rank, so with
        // Pk == tp, group kk maps one-to-one onto shard kk and exactly ONE group can make progress per
        // arrival wave -- every other group sits idle through fabric startup.
        //
        // Instead give every group a stripe of EVERY shard. Wave 0 (the local shard) then hands all Pk
        // groups work immediately, and each subsequent wave (two shards, one per direction) does the same.
        // The group's capacity-local index l maps to a global K tile as
        //     gk(l) = (l / run_len) * shard_stride + stripe_base + (l % run_len)
        // which both the in0 side and the in1 reader evaluate identically -- the spec requires the in1
        // reader to walk the same global-K order, and this is that order.
        //
        // The per-shard split is balanced, not exact-divide, so a shard tile count that does not divide by
        // Pk still distributes rather than being refused; run_len is then constant per group across shards
        // (every shard has the same tile count), which keeps the closed form above valid.
        uint32_t k_run_len = 0u, k_stripe_base = 0u, k_shard_stride = 0u, k_valid = cp.valid_k;
        // Capacity-local slots reserved per source rank. Equal to k_run_len on the staged path (rank stripes
        // are packed back to back); under direct-L1 each rank is padded up to a whole number of ring slots so
        // one slot never straddles two ranks -- see build_direct_l1_plan.
        uint32_t k_rank_span = 0u;
        if (fused_gather.enabled) {
            k_shard_stride = geo.Kt / fused_gather.tp;  // tiles per source rank (Kt % tp == 0 asserted below)
            const plan::BalRange rs = plan::rap_balanced(cp.kk, k_shard_stride, Pk);
            k_run_len = rs.extent;
            k_stripe_base = rs.start;
            k_rank_span = direct_l1 ? dl1_plan[i].rank_span : k_run_len;
            k_valid = k_rank_span * fused_gather.tp;
            TT_FATAL(
                k_run_len > 0u,
                "the blocked-cyclic K assignment gives Pk group {} no tiles: Pk={} exceeds the {} tiles per "
                "source rank. Reduce Pk or increase K",
                cp.kk,
                Pk,
                k_shard_stride);
            TT_FATAL(
                k_valid <= geo.K_slice_capacity,
                "blocked-cyclic K needs {} tiles for Pk group {} but the planner sized the slice capacity at "
                "{}; the per-shard rounding has outgrown the contiguous estimate",
                k_valid,
                cp.kk,
                geo.K_slice_capacity);
        }

        // in1 reader runtime args.
        std::vector<uint32_t> ra = {
            in1_addr,     // 0
            cp.bank,      // 1
            cp.ring_pos,  // 2
            cp.k_start,   // 3 first logical K tile (balanced)
            cp.n_local,   // 4 within-bank column offset
            k_valid,      // 5 valid K tiles (rest of capacity zero-filled)
            cp.valid_n};  // 6 valid N tiles this core owns
        // 7..7+G-1: the consume order, from the SAME RingSchedule the writer's slot args come from. The
        // reader used to recompute it as (ring_pos + G - step) % G, i.e. a second independent derivation of
        // the global-K order the spec requires both sides to walk identically. One array, one derivation.
        for (uint32_t s = 0; s < geo.G; ++s) {
            ra.push_back(ring_sch[cp.slice].consume_pos[cp.ring_pos][s]);
        }
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
        if (fused_gather.enabled) {
            // Appended last, after the consume order and the variable-length M-split peer coords; the kernel
            // locates them at 9 + G + 2*mpeers. Only present when FUSED_GATHER is defined for the reader.
            ra.push_back(k_run_len);
            ra.push_back(k_stripe_base);
            ra.push_back(k_shard_stride);
            ra.push_back(k_rank_span);
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
            k_valid,                 // 14 valid K tiles (rest of capacity zero)
            cp.valid_m,              // 15 valid M tiles (rest zero / not written)
            cp.valid_n};             // 16 valid N tiles (rest zero / not written)
        // ---- Ring slot schedule (args 17..32; see RingSlotArg) ----
        // PHASE 2: derived from `ring_sch` rather than hand-written, so the writer and the in1 reader cannot
        // describe different orders. Still today's rotation, so the program stays bit-identical.
        {
            const RingSchedule& sch = ring_sch[cp.slice];
            const uint32_t p = cp.ring_pos;
            const uint32_t succ = (p + 1u) % geo.G;
            const uint32_t own_slot = sch.slot_of(p, p);  // the step at which I consume my own stripe
            // Where each NEIGHBOUR puts this same chunk. With one shared schedule that is just own_slot; with
            // availability order the neighbours sort differently, so ask their schedules directly.
            const uint32_t peer_fwd = ring_sch_fwd.empty() ? own_slot : ring_sch_fwd[cp.slice].slot_of(p, p);
            const uint32_t peer_bwd = ring_sch_bwd.empty() ? own_slot : ring_sch_bwd[cp.slice].slot_of(p, p);
            wa.push_back(own_slot);
            wa.push_back(peer_fwd);
            wa.push_back(peer_bwd);
            // FORWARD ORDER = my consume order with my SUCCESSOR'S OWN chunk removed (it already has that
            // one, and 7 forwards cover the other 7). Because forwarding follows availability order too, a
            // core waiting on its own late chunk keeps relaying the ones behind it -- the coupling that made
            // one late chunk freeze the whole ring.
            for (uint32_t s = 0, t = 0; s < geo.G; ++s) {
                const uint32_t chunk = sch.consume_pos[p][s];
                if (chunk == succ) {
                    continue;  // never forward the successor's own chunk
                }
                TT_FATAL(t + 1u < geo.G, "forward order overflow at position {}", p);
                wa.push_back(sch.slot_of(p, chunk));     // read this slot of mine
                wa.push_back(sch.slot_of(succ, chunk));  // write that slot on my successor
                ++t;
            }
            // The ROTATION must still come out exactly as phase 1 hand-derived it: own_slot 0 and
            // (src, dst) == (s, s+1). Keeps the staged path (and TT_AGMM_RING_ROTATION=1) bit-identical, and
            // catches a schedule generator that has broken the case we can check exactly.
            if (!direct_l1 || ring_rotation) {
                TT_FATAL(own_slot == 0u, "rotation must give own_slot 0, got {}", own_slot);
                for (uint32_t s = 0; s + 1u < geo.G; ++s) {
                    TT_FATAL(
                        wa[kRsFwdBase + 2u * s] == s && wa[kRsFwdBase + 2u * s + 1u] == s + 1u,
                        "rotation must reproduce (s, s+1) at position {} step {}: got ({}, {})",
                        p,
                        s,
                        wa[kRsFwdBase + 2u * s],
                        wa[kRsFwdBase + 2u * s + 1u]);
                }
            }
            TT_FATAL(
                wa.size() == kRsArgCount,
                "ring slot schedule ends at writer arg {} but RingSlotArg says {}; the push order and the "
                "kernel's indices have drifted",
                wa.size(),
                static_cast<uint32_t>(kRsArgCount));
        }
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

        // ---- PHASE 1 fused fabric all-gather args (appended last; only on the MASTER ring) ----
        // Emitted for the 8 cores of ring group p == 0 -- the only fabric clients. Every other core reads
        // the staged shards but never sends, so it needs no mux connection and gets none of these.
        //
        // Layout: [rank, tp, k_shard_tiles, stage_addr, chunk_ready_sem_id, has_fwd, has_bwd]
        //         then, per present direction, the mux client-connection block appended by
        //         append_client_connection_rt_args (opaque to us -- the kernel hands it to FabricMuxV2Sender).
        if (fused_gather.enabled) {
            TT_FATAL(
                wa.size() == fused_rt_base,
                "fused_rt_base ({}) must match the actual writer arg count ({}); the CT arg and the wa push "
                "order have drifted",
                fused_rt_base,
                wa.size());
            // EVERY core gets the common block at the same index (`fused_rt_base`, a compile-time arg), so
            // the kernel can locate it without knowing which optional fusion args preceded it. The first
            // word says whether this core is a fabric client; only master-ring cores get the mux block.
            const bool is_master_ring = (i % preaders_pf) == 0u;
            // Kt must divide evenly by tp, and each shard must be tile-aligned. Two distinct silent
            // corruptions otherwise: the kernel strides the local shard as m * k_shard_tiles, which is
            // only the true row stride when k_local is tile-aligned; and when Kt % tp != 0 the staging
            // columns in [tp*k_shard_tiles, Kt) are never written but ARE read by the matmul.
            TT_FATAL(
                geo.Kt % fused_gather.tp == 0,
                "fused gather needs the global K tile count ({}) to divide by tp ({}); the remainder "
                "columns would be read by the matmul but never staged",
                geo.Kt,
                fused_gather.tp);
            const uint32_t k_local_elems = tensor_args.input_tensor.logical_shape()[-1];
            TT_FATAL(
                k_local_elems % tt::constants::TILE_WIDTH == 0,
                "fused gather needs each K shard tile-aligned, got a {}-element shard (TILE_WIDTH={}); the "
                "kernel would stride the local shard by the wrong row pitch",
                k_local_elems,
                tt::constants::TILE_WIDTH);
            const uint32_t k_shard_tiles = geo.Kt / fused_gather.tp;  // this rank's K tiles
            wa.push_back(is_master_ring ? 1u : 0u);
            wa.push_back(fused_gather.rank);
            wa.push_back(fused_gather.tp);
            wa.push_back(k_shard_tiles);
            wa.push_back(tensor_args.gather_staging_buffer->buffer()->address());
            wa.push_back(fused_gather.chunk_ready_sem_id);
            // Direction split: the first half of the masters drive forward, the second half backward.
            // Each direction's cores between them cover all of M, so a core's fabric M range is indexed by
            // (bank_id mod m_groups), not by bank_id.
            const uint32_t m_groups_pf = fused_gather.num_masters / 2u;
            const uint32_t bank_pf = i / preaders_pf;
            const bool core_is_bwd = is_master_ring && (bank_pf >= m_groups_pf);
            // Under direct-L1 the client set is the whole compute grid and each core's direction comes from
            // its stream plan, not from the master-ring halves. A LINE origin drives BOTH muxes, which is the
            // one case where a single core registers twice.
            const bool has_fwd_client = direct_l1
                                            ? (dl1_plan[i].send_fwd && !fused_gather.mux_cfgs_fwd.empty())
                                            : (is_master_ring && !core_is_bwd && !fused_gather.mux_cfgs_fwd.empty());
            const bool has_bwd_client = direct_l1
                                            ? (dl1_plan[i].send_bwd && !fused_gather.mux_cfgs_bwd.empty())
                                            : (is_master_ring && core_is_bwd && !fused_gather.mux_cfgs_bwd.empty());
            wa.push_back(has_fwd_client ? 1u : 0u);
            wa.push_back(has_bwd_client ? 1u : 0u);
            wa.push_back(Mt_r);                          // global M tiles (staging row count)
            wa.push_back(geo.Kt);                        // global K tiles (staging row stride)
            wa.push_back(in0.buffer()->address());       // LOCAL shard base (in0_addr now points at staging)
            wa.push_back(bank_pf);                       // bank id 0..7 -> which M-slice this core stages locally
            // My direction's credit counter, and the round split. fwd = ceil((tp-1)/2), bwd = the rest, so
            // they sum to exactly tp-1: at even tp the antipode rides the forward stream only and is
            // therefore delivered exactly once.
            // RING: the two directions split tp-1 evenly (tp is validated even, so fwd = tp/2), and each
            // core sends exactly as many shards as it receives.
            // LINE: counts are per-rank and send != recv. Node d passes forward the d+1 shards that
            // originated at 0..d and receives the d shards 0..d-1; symmetrically backward. The two receive
            // counts still sum to tp-1, so every rank still ends up with the whole activation. A rank at
            // an end has no mux in that direction and sends nothing there, but it still RECEIVES the full
            // tp-1 from the one direction it does have.
            const uint32_t rk = fused_gather.rank;
            uint32_t fwd_send, fwd_recv, bwd_send, bwd_recv;
            if (operation_attributes.topology_is_ring) {
                fwd_send = fwd_recv = fused_gather.tp / 2u;
                bwd_send = bwd_recv = (fused_gather.tp - 1u) - fwd_send;
            } else {
                fwd_send = !fused_gather.mux_cfgs_fwd.empty() ? (rk + 1u) : 0u;
                fwd_recv = rk;
                bwd_send = !fused_gather.mux_cfgs_bwd.empty() ? (fused_gather.tp - rk) : 0u;
                bwd_recv = fused_gather.tp - 1u - rk;
            }
            wa.push_back(core_is_bwd ? fused_gather.bwd_recv_sem_addr : fused_gather.fwd_recv_sem_addr);
            wa.push_back(core_is_bwd ? 1u : 0u);
            wa.push_back(core_is_bwd ? bwd_send : fwd_send);
            wa.push_back(core_is_bwd ? bwd_recv : fwd_recv);
            wa.push_back(m_groups_pf);
            wa.push_back((is_master_ring && (i == 0u)) ? 1u : 0u);
            wa.push_back(fused_gather.gather_count_sem_id);
            wa.push_back(fused_gather.num_masters);
            wa.push_back(static_cast<uint32_t>(fused_gather.master0_virtual.x));
            wa.push_back(static_cast<uint32_t>(fused_gather.master0_virtual.y));
            wa.push_back(fused_gather.dir_count_sem_id);
            wa.push_back(fused_gather.wave_fwd_sem_id);
            wa.push_back(fused_gather.wave_bwd_sem_id);
            wa.push_back(static_cast<uint32_t>(fused_gather.fwd_coord_virtual.x));
            wa.push_back(static_cast<uint32_t>(fused_gather.fwd_coord_virtual.y));
            wa.push_back(static_cast<uint32_t>(fused_gather.bwd_coord_virtual.x));
            wa.push_back(static_cast<uint32_t>(fused_gather.bwd_coord_virtual.y));
            wa.push_back(fwd_recv);
            wa.push_back(bwd_recv);
            wa.push_back(fused_gather.local_done_sem_id);
            wa.push_back(k_run_len);
            wa.push_back(k_stripe_base);
            wa.push_back(k_shard_stride);
            wa.push_back(fused_gather.fwd_coord_swaps);
            wa.push_back(fused_gather.bwd_coord_swaps);
            wa.push_back(k_rank_span);
            // ---- direct-L1 stream plan (all zero on the staged path) ----
            // One arrival semaphore per core is the entire synchronisation. It is a GLOBAL semaphore, not a
            // program one, for the same reason the staged path's credit is: the upstream device can credit us
            // before our program has launched, and launch zeroes program semaphores.
            wa.push_back((direct_l1 && dl1_plan[i].has_stripe) ? 1u : 0u);
            wa.push_back(direct_l1 ? dl1_plan[i].dist : 0u);
            wa.push_back((direct_l1 && dl1_plan[i].send_fwd) ? 1u : 0u);
            wa.push_back((direct_l1 && dl1_plan[i].send_bwd) ? 1u : 0u);
            wa.push_back(direct_l1 ? operation_attributes.gather_semaphores[0].address() : 0u);
            wa.push_back(fused_gather.dl1_packet_bytes);
            // LEAVES ONLY. A relay cannot defer as the kernel stands: its prologue SENDS its chunk onward, so
            // skipping the wait ships garbage -- measured directly, 12/12 phase-gate failures. Deferring a
            // relay additionally requires moving its fabric send to arrival-time, which needs the mux
            // open/close hoisted out of the prologue. Until then this is a leaf-only optimisation.
            wa.push_back(
                (direct_l1 && dl1_plan[i].has_stripe && dl1_plan[i].dist != 0u && !dl1_plan[i].send_fwd &&
                 !dl1_plan[i].send_bwd)
                    ? 1u
                    : 0u);
            wa.push_back(static_cast<uint32_t>(fused_gather.release_ranges.size()));
            for (const auto& rr : fused_gather.release_ranges) {
                wa.push_back(rr.sx);
                wa.push_back(rr.sy);
                wa.push_back(rr.ex);
                wa.push_back(rr.ey);
                wa.push_back(rr.dests_fwd);
                wa.push_back(rr.dests_bwd);
            }
            for (const auto& mv : fused_gather.master_virtuals) {
                wa.push_back(static_cast<uint32_t>(mv.x));
                wa.push_back(static_cast<uint32_t>(mv.y));
            }
            TT_FATAL(
                wa.size() == fused_rt_base + kFusedArgCount + 6u * fused_gather.release_ranges.size() +
                                 2u * fused_gather.master_virtuals.size(),
                "fused-gather arg block is {} words but FusedGatherArg says {} + 6*{}; the push order and "
                "the offsets used by override_runtime_arguments have drifted",
                wa.size() - fused_rt_base,
                static_cast<uint32_t>(kFusedArgCount),
                fused_gather.release_ranges.size());
            // Register ONLY with the mux this core actually drives. Registering with both would hand each
            // mux clients that never open or close it, and mux v2 terminates by counting close() calls --
            // its forwarder RISC would spin forever.
            if (has_fwd_client) {
                const auto fc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                const auto tc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                // Deal clients round-robin across this direction's links, so consecutive senders use
                // different links rather than piling onto one. Direct-L1 keys the round robin off a running
                // counter (its client set is every core, not one per bank) -- and that counter has to advance
                // in exactly the pattern mux_channel_counts assumed when it sized each mux.
                const uint32_t Lf = direct_l1 ? (fused_gather.next_link_fwd++ % fused_gather.mux_cfgs_fwd.size())
                                              : (bank_pf % fused_gather.num_links);
                fused_gather.mux_cfgs_fwd[Lf]->append_client_connection_rt_args(
                    fused_gather.mux_virtual_cores_fwd[Lf],
                    fused_gather.next_channel_fwd[Lf]++,
                    tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{fc, tc},
                    wa);
            }
            if (has_bwd_client) {
                const auto fc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                const auto tc = CreateSemaphore(program, CoreRangeSet(CoreRange(cores[i], cores[i])), 0);
                const uint32_t Lb = direct_l1 ? (fused_gather.next_link_bwd++ % fused_gather.mux_cfgs_bwd.size())
                                              : ((bank_pf - m_groups_pf) % fused_gather.num_links);
                fused_gather.mux_cfgs_bwd[Lb]->append_client_connection_rt_args(
                    fused_gather.mux_virtual_cores_bwd[Lb],
                    fused_gather.next_channel_bwd[Lb]++,
                    tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{fc, tc},
                    wa);
            }
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

    // Every mux channel must have been claimed by a client that will really open and close it. Mux v2
    // self-terminates by counting close() calls against its compile-time channel count, so a mismatch is a
    // HANG with no diagnostic, not an error -- this branch's history has three of them. Cheap to assert here,
    // impossible to attribute later.
    if (fused_gather.enabled) {
        for (uint32_t L = 0; L < fused_gather.mux_cfgs_fwd.size(); ++L) {
            TT_FATAL(
                fused_gather.next_channel_fwd[L] == fused_gather.mux_channels_fwd[L],
                "forward mux {} was sized for {} channels but {} clients registered; mux v2 waits for one "
                "close() per channel, so this would hang",
                L,
                fused_gather.mux_channels_fwd[L],
                fused_gather.next_channel_fwd[L]);
        }
        for (uint32_t L = 0; L < fused_gather.mux_cfgs_bwd.size(); ++L) {
            TT_FATAL(
                fused_gather.next_channel_bwd[L] == fused_gather.mux_channels_bwd[L],
                "backward mux {} was sized for {} channels but {} clients registered; mux v2 waits for one "
                "close() per channel, so this would hang",
                L,
                fused_gather.mux_channels_bwd[L],
                fused_gather.next_channel_bwd[L]);
        }
    }

    return ttnn::device_operation::CachedProgram<shared_variables_t>{
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
            .n_chunks = n_chunks,
            .fused_gather = fused_gather.enabled,
            .fused_rt_base = fused_rt_base,
            .preaders = preaders_pf,
            .fwd_master_count = fused_gather.enabled ? (fused_gather.num_masters / 2u) : 0u,
            .direct_l1 = direct_l1}};
}

AllGatherRegimeAMatmulAsyncProgramFactory::cached_mesh_workload_t
AllGatherRegimeAMatmulAsyncProgramFactory::create_mesh_workload(
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    for (auto& [_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(_range);

        // Must track the SAME tensor create_at bound: the staging buffer on the fused path.
        const uint32_t in0_addr = (operation_attributes.tp > 1) ? tensor_args.gather_staging_buffer->buffer()->address()
                                                                : tensor_args.input_tensor.buffer()->address();
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
            uint32_t fidx = kRsArgCount;  // optional args start after the fixed prefix + ring slot schedule
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

            // ---- fused-gather block ----
            // Everything the caller ping-pongs lives here, so it MUST be refreshed on every replay. The
            // staging buffer and the global semaphore rotate between invocations by design (that is what
            // makes repeated CCL safe), which means a cached program that kept the first invocation's
            // addresses would gather into a buffer nobody reads and wait on a credit nobody sends.
            // Offsets mirror the push order in create_at; fused_rt_base is captured there.
            if (sv.fused_gather) {
                const uint32_t g = sv.fused_rt_base;
                wa[g + kFgStageAddr] = tensor_args.gather_staging_buffer->buffer()->address();
                wa[g + kFgShardAddr] = tensor_args.input_tensor.buffer()->address();
                // Per-direction credit. This MUST mirror create_at's split exactly: a replay that gets it
                // wrong hands a core the other direction's counter, so it waits on credits that arrive
                // somewhere else -- and only on a cached program, i.e. from the second invocation on.
                // The threshold is carried in shared_variables rather than recomputed from a literal,
                // because widening the fabric client set past 8 masters is the next planned change and a
                // hardcoded 4 here would desync silently.
                const bool is_bwd = ((i / sv.preaders) >= sv.fwd_master_count);
                wa[g + kFgMyRecvSem] = operation_attributes.gather_semaphores[is_bwd ? 1u : 0u].address();
                // Direct-L1's arrival semaphore rotates with the caller's set exactly like the staged
                // credit does. Leaving this stale is the classic "correct on invocation 1, wedged from 2 on"
                // failure: every non-origin core would wait on a word nobody increments. Only patched when
                // the core actually has one (0 means "no stream", and must stay 0).
                if (sv.direct_l1 && wa[g + kFgDl1Active] != 0u) {
                    wa[g + kFgDl1RecvSem] = operation_attributes.gather_semaphores[0].address();
                }
            }
        }
    }  // per-coordinate program
}

}  // namespace ttnn::experimental::prim
