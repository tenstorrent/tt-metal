// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_program_factory.hpp"

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

// Tile-byte sizes are defined once in regime_a_matmul_plan.hpp (single source of truth), reached via `plan::`.
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

// TEST-ONLY (diag bit9 RING_BALANCED): LINK-LOAD-BALANCED in0 ring ordering. Correctness-preserving and
// host-only — same ring MEMBERSHIP as production (the 8 banks of a slice), only the visiting order changes.
//
// Why: `optimize_in0_ring_order` minimizes each ring's hop cost INDEPENDENTLY, so all Pk*Ns rings converge
// on the same shortest corridors and pile up on the same links. An offline route model (validated exactly
// against get_worker_noc_hop_distance, see tools/mm_sweep/picker_gen/ring_topology_probe.py) shows the wall
// tracks the BUSIEST LINK, not total hops: a compact re-partitioning cut total hops 16% but raised the
// busiest link 25% and measured 8.7% SLOWER, while re-ordering alone can cut the busiest link 25% at
// unchanged hops. Every ring edge carries 7 shards, so peak link load is what the 7 serial steps pay for.
//
// Objective: process groups sequentially (two passes), and for each pick the cycle minimizing the peak
// GLOBAL link load it contributes to (tie-break: total hops), given the loads already committed. Each
// candidate's load counts all Sm mm-rings, since one permutation is shared by the whole group.
//
// Route model: a single coordinate frame reconstructed from the device (per-logical-step hop distances +
// the wrap-around distance give the physical spacing and torus extent), NOC_0 travelling +x/+y and NOC_1
// -x/-y. The dimension ORDER (x-first vs y-first) is not observable from hop counts, so each edge is
// charged on BOTH candidate routes — the balancing then spreads load whichever order the hardware uses.
class RingLinkModel {
public:
    RingLinkModel(IDevice* device, uint32_t gx, uint32_t gy) : px_(gx, 0), py_(gy, 0) {
        namespace expd = tt::tt_metal::experimental::Device;
        auto h = [&](CoreCoord a, CoreCoord b) {
            return expd::get_worker_noc_hop_distance(device, a, b, NOC::NOC_0);  // NOC_0 == +x/+y
        };
        for (uint32_t i = 0; i + 1 < gx; ++i) {
            px_[i + 1] = px_[i] + h(CoreCoord{i, 0}, CoreCoord{i + 1, 0});
        }
        for (uint32_t j = 0; j + 1 < gy; ++j) {
            py_[j + 1] = py_[j] + h(CoreCoord{0, j}, CoreCoord{0, j + 1});
        }
        wx_ = px_[gx - 1] + h(CoreCoord{gx - 1, 0}, CoreCoord{0, 0});  // torus extents incl. non-worker rows/cols
        wy_ = py_[gy - 1] + h(CoreCoord{0, gy - 1}, CoreCoord{0, 0});
        nlinks_ = 4u * wx_ * wy_;  // (noc, axis) x position
    }

    uint32_t num_links() const { return nlinks_; }
    // Relative-frame coordinates. physical == (px+1, py+2) on BH; the DRAM columns are therefore at
    // relative x = px(0)-1 (banks 0-3, physical x=0) and px(5)+3 (banks 4-7, physical x=9) — both confirmed
    // by the bank-adjacent worker assignment and blackhole_140_arch.yaml.
    uint32_t rx(uint32_t lx) const { return px_[lx]; }
    uint32_t ry(uint32_t ly) const { return py_[ly]; }
    uint32_t dram_rx(uint32_t bank) const { return (bank < 4u) ? ((px_[0] + wx_ - 1u) % wx_) : (px_[5] + 3u); }

    // Distinct links traversed from `s` to `d` on `noc`, charging BOTH dimension orders.
    void links(const plan::PlanXY& s, const plan::PlanXY& d, uint32_t noc, std::vector<uint32_t>& out) const {
        links_rel(px_[s.x], py_[s.y], px_[d.x], py_[d.y], noc, out);
    }

    void links_rel(uint32_t sx0, uint32_t sy0, uint32_t tx, uint32_t ty, uint32_t noc, std::vector<uint32_t>& out)
        const {
        out.clear();
        for (uint32_t order = 0; order < 2u; ++order) {
            uint32_t x = sx0, y = sy0;
            const uint32_t sx = (noc == 0u) ? 1u : (wx_ - 1u);  // +1 / -1 modulo the torus extent
            const uint32_t sy = (noc == 0u) ? 1u : (wy_ - 1u);
            auto walk_x = [&](uint32_t& x, uint32_t y) {
                while (x != tx) {
                    push_unique(out, id(noc, 0u, x, y));
                    x = (x + sx) % wx_;
                }
            };
            auto walk_y = [&](uint32_t x, uint32_t& y) {
                while (y != ty) {
                    push_unique(out, id(noc, 1u, x, y));
                    y = (y + sy) % wy_;
                }
            };
            if (order == 0u) {
                walk_x(x, y);
                walk_y(x, y);
            } else {
                walk_y(x, y);
                walk_x(x, y);
            }
        }
    }

private:
    static void push_unique(std::vector<uint32_t>& v, uint32_t e) {
        if (std::find(v.begin(), v.end(), e) == v.end()) {
            v.push_back(e);
        }
    }
    uint32_t id(uint32_t noc, uint32_t axis, uint32_t x, uint32_t y) const {
        return ((noc * 2u + axis) * wx_ + x) * wy_ + y;
    }
    uint32_t wx_{1}, wy_{1}, nlinks_{0};
    std::vector<uint32_t> px_, py_;
};

// TEST-ONLY (diag bit8 RING_REGIONAL): REGION-LOCAL in0 ring re-partitioning. Correctness-preserving and
// host-only — no kernel define, no extra runtime arg, output still valid.
//
// The in0 ring is purely an in0-DELIVERY construct: every core sharing a given (kk, mm) needs the SAME
// k-slice, INCLUDING the Ns cores that differ only in n-slice index nn. Today a ring is hardcoded to "the 8
// banks of one slice j", and on BH the 8 bank-adjacent workers sit in just two columns (x=0 for banks 0-3,
// x=6 for banks 4-7), so EVERY ring straddles that ~6-hop bisection twice and each crossing edge carries 7
// shards. D1 attribution measured ~70% of the ring-forward cost to be hop distance, so that geometry is the
// dominant term. Here we instead partition the 8*Ns cores of each (kk, mm) group into Ns PHYSICALLY COMPACT
// rings of 8 (for Ns=2 that is exactly 8 left / 8 right, i.e. zero bisection crossings).
//
// Unchanged by construction: placement, bank adjacency, in0 DRAM read VOLUME (each ring still reads the
// whole slice once, so Ns copies as before), the in1 read, the reduction chain, CBs, semaphores and all
// three kernels. Only ring_pos / ring_next_idx / ring_prev_idx move, and the ring protocol is correct for
// ANY permutation of any 8 same-(kk,mm) cores.
//
// Rings become MIXED-NoC (nn selects the writer NoC), which is functionally fine — payload-then-semaphore
// ordering only needs the two to share a sender+NoC — so every edge is costed on the SENDER's writer NoC.
//
// M-split: mm-siblings consume the in1 stream in the mm==0 reader's shard order, so all mm MUST share one
// (bank,nn) -> ring_pos map. The partition and the cyclic order are therefore computed ONCE per kk on costs
// aggregated over the Sm mm-rings, then mirrored to every mm.
void regroup_in0_rings(
    plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo, uint32_t Pk, uint32_t Ns, uint32_t Sm) {
    namespace expd = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    const uint32_t mfac = geo.mfac;  // Ns*Sm
    const uint32_t nitems = 8u * Ns;
    // item k <-> (bank, nn); core index of item k for a given (kk, mm).
    auto core_idx = [&](uint32_t kk, uint32_t k, uint32_t mm) {
        return (k / Ns) * preaders + (kk * mfac + (k % Ns) * Sm + mm);
    };

    for (uint32_t kk = 0; kk < Pk; ++kk) {
        // Directed per-mm hop matrices over the items, each edge on the SENDER's writer NoC.
        std::vector<std::vector<std::vector<uint32_t>>> dm(
            Sm, std::vector<std::vector<uint32_t>>(nitems, std::vector<uint32_t>(nitems, 0u)));
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t a = 0; a < nitems; ++a) {
                const auto& ca = P.cores[core_idx(kk, a, mm)];
                const NOC wnoc = (ca.noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;  // sender's writer NoC
                for (uint32_t b = 0; b < nitems; ++b) {
                    if (a == b) {
                        continue;
                    }
                    const auto& cb = P.cores[core_idx(kk, b, mm)];
                    dm[mm][a][b] = expd::get_worker_noc_hop_distance(
                        device, CoreCoord{ca.coord.x, ca.coord.y}, CoreCoord{cb.coord.x, cb.coord.y}, wnoc);
                }
            }
        }
        // Symmetric proximity summed over the mm-rings (clustering metric only).
        auto sym = [&](uint32_t a, uint32_t b) {
            uint32_t s = 0;
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                s += dm[mm][a][b] + dm[mm][b][a];
            }
            return s;
        };

        // --- Partition into Ns compact groups of 8. Greedy: the most PERIPHERAL unassigned item seeds a
        // group (so a group never starts in the middle and straddles), then the 7 nearest-to-the-group
        // unassigned items join it. On the BH two-column geometry with Ns=2 this yields exactly the
        // left-region and right-region groups.
        constexpr uint32_t kUnset = 0xffffffffu;
        std::vector<uint32_t> assigned(nitems, kUnset);
        std::vector<std::array<uint32_t, 8>> groups;
        for (uint32_t g = 0; g < Ns; ++g) {
            uint32_t seed = 0;
            uint64_t seed_score = 0;
            bool have_seed = false;
            for (uint32_t a = 0; a < nitems; ++a) {
                if (assigned[a] != kUnset) {
                    continue;
                }
                uint64_t s = 0;
                for (uint32_t b = 0; b < nitems; ++b) {
                    if (b != a && assigned[b] == kUnset) {
                        s += sym(a, b);
                    }
                }
                if (!have_seed || s > seed_score) {
                    seed_score = s;
                    seed = a;
                    have_seed = true;
                }
            }
            std::array<uint32_t, 8> grp{};
            grp[0] = seed;
            assigned[seed] = g;
            for (uint32_t n = 1; n < 8u; ++n) {
                uint32_t best = 0;
                uint64_t bestd = 0;
                bool found = false;
                for (uint32_t a = 0; a < nitems; ++a) {
                    if (assigned[a] != kUnset) {
                        continue;
                    }
                    uint64_t d = ~0ull;  // nearest-to-group (single link) keeps the blob connected
                    for (uint32_t m = 0; m < n; ++m) {
                        d = std::min<uint64_t>(d, sym(a, grp[m]));
                    }
                    if (!found || d < bestd) {
                        bestd = d;
                        best = a;
                        found = true;
                    }
                }
                grp[n] = best;
                assigned[best] = g;
            }
            groups.push_back(grp);
        }

        // --- Cyclic order per group: the SAME two-pass PARETO objective as optimize_in0_ring_order (pass 1
        // scores the mm==0 ring to establish the aggtot budget and seed; pass 2 minimizes the worst directed
        // edge over all mm-rings subject to that budget), so ordering quality is directly comparable.
        struct Metrics {
            uint32_t r0max, r0tot, aggmax, aggtot;
        };
        auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) { return a0 < b0 || (a0 == b0 && a1 < b1); };
        for (uint32_t g = 0; g < Ns; ++g) {
            const std::array<uint32_t, 8>& items = groups[g];
            auto metrics = [&](const std::array<uint32_t, 8>& ord) -> Metrics {
                Metrics m{0, 0, 0, 0};
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    uint32_t mx = 0, tot = 0;
                    for (uint32_t p = 0; p < 8u; ++p) {
                        const uint32_t e = dm[mm][ord[p]][ord[(p + 1u) % 8u]];
                        tot += e;
                        mx = std::max(mx, e);
                    }
                    if (mm == 0) {
                        m.r0max = mx;
                        m.r0tot = tot;
                    }
                    m.aggmax = std::max(m.aggmax, mx);
                    m.aggtot += tot;
                }
                return m;
            };
            // fix items[0] at position 0 and permute the other 7 (5040 directed cycles).
            auto cand_of = [&](const std::array<uint32_t, 7>& t) {
                std::array<uint32_t, 8> c{};
                c[0] = items[0];
                for (uint32_t i = 0; i < 7u; ++i) {
                    c[i + 1u] = t[i];
                }
                return c;
            };
            std::array<uint32_t, 7> tail{};
            for (uint32_t i = 0; i < 7u; ++i) {
                tail[i] = items[i + 1u];
            }
            std::sort(tail.begin(), tail.end());
            std::array<uint32_t, 8> opt_mm0 = cand_of(tail);
            Metrics b_mm0 = metrics(opt_mm0);
            std::array<uint32_t, 7> t1 = tail;
            do {
                const std::array<uint32_t, 8> cand = cand_of(t1);
                const Metrics m = metrics(cand);
                if (lt2(m.r0max, m.r0tot, b_mm0.r0max, b_mm0.r0tot)) {
                    b_mm0 = m;
                    opt_mm0 = cand;
                }
            } while (std::next_permutation(t1.begin(), t1.end()));
            std::array<uint32_t, 8> opt = opt_mm0;
            Metrics b_pa = b_mm0;
            const uint32_t budget = b_mm0.aggtot;
            std::array<uint32_t, 7> t2 = tail;
            do {
                const std::array<uint32_t, 8> cand = cand_of(t2);
                const Metrics m = metrics(cand);
                if (m.aggtot <= budget && lt2(m.aggmax, m.aggtot, b_pa.aggmax, b_pa.aggtot)) {
                    b_pa = m;
                    opt = cand;
                }
            } while (std::next_permutation(t2.begin(), t2.end()));

            // --- Apply the same (bank,nn) -> ring_pos map to every mm (M-split in1 pairing requirement).
            for (uint32_t mm = 0; mm < Sm; ++mm) {
                for (uint32_t pos = 0; pos < 8u; ++pos) {
                    const uint32_t ci = core_idx(kk, opt[pos], mm);
                    P.cores[ci].ring_pos = pos;
                    P.cores[ci].ring_next_idx = core_idx(kk, opt[(pos + 1u) % 8u], mm);
                    P.cores[ci].ring_prev_idx = core_idx(kk, opt[(pos + 7u) % 8u], mm);
                }
            }
        }
    }
}

// Link-load-balanced ring ordering (see RingLinkModel above for the objective + route model). Same ring
// membership as production; only ring_pos / ring_next_idx / ring_prev_idx change, and the ring protocol is
// correct for any permutation. One permutation per (kk,nn) group, shared by its Sm mm-rings, exactly as
// optimize_in0_ring_order does (the M-split in1 pairing requires it).
void balance_in0_ring_order(
    plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo, uint32_t Sm, const CoreCoord& grid) {
    namespace expd = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    const RingLinkModel lm(device, grid.x, grid.y);
    std::vector<uint32_t> load(lm.num_links(), 0u);

    const uint32_t ngroups = preaders / Sm;
    // Per group + mm: the 8x8 edge -> link-list and hop-cost tables (bank indices).
    struct GroupTables {
        uint32_t base{};
        NOC wnoc{};
        std::vector<std::array<std::array<std::vector<uint32_t>, 8>, 8>> lk;  // [mm][a][b]
        std::vector<std::array<std::array<uint32_t, 8>, 8>> hp;               // [mm][a][b]
    };
    std::vector<GroupTables> gt(ngroups);
    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t base = g * Sm;
        gt[g].base = base;
        gt[g].wnoc = (P.cores[base].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;  // shared writer NoC of the group
        const uint32_t wnoc_idx = (gt[g].wnoc == NOC::NOC_0) ? 0u : 1u;
        gt[g].lk.resize(Sm);
        gt[g].hp.resize(Sm);
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t a = 0; a < 8u; ++a) {
                const auto& ca = P.cores[a * preaders + base + mm].coord;
                for (uint32_t b = 0; b < 8u; ++b) {
                    if (a == b) {
                        gt[g].hp[mm][a][b] = 0u;
                        continue;
                    }
                    const auto& cb = P.cores[b * preaders + base + mm].coord;
                    lm.links(ca, cb, wnoc_idx, gt[g].lk[mm][a][b]);
                    gt[g].hp[mm][a][b] = expd::get_worker_noc_hop_distance(
                        device, CoreCoord{ca.x, ca.y}, CoreCoord{cb.x, cb.y}, gt[g].wnoc);
                }
            }
        }
    }

    // Sequential greedy, two passes: re-choose each group's cycle against the loads committed by the others.
    std::vector<std::array<uint32_t, 8>> chosen(ngroups);
    std::vector<bool> placed(ngroups, false);
    std::vector<uint32_t> touched;  // scratch: links of the candidate under evaluation
    auto apply = [&](uint32_t g, const std::array<uint32_t, 8>& ord, int delta) {
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t p = 0; p < 8u; ++p) {
                for (const uint32_t l : gt[g].lk[mm][ord[p]][ord[(p + 1u) % 8u]]) {
                    load[l] = static_cast<uint32_t>(static_cast<int>(load[l]) + delta);
                }
            }
        }
    };
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        for (uint32_t g = 0; g < ngroups; ++g) {
            if (placed[g]) {
                apply(g, chosen[g], -1);
            }
            std::array<uint32_t, 8> best{0, 1, 2, 3, 4, 5, 6, 7};
            uint32_t best_peak = ~0u, best_hops = ~0u;
            std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
            do {
                std::array<uint32_t, 8> cand{};
                cand[0] = 0u;
                for (uint32_t i = 0; i < 7u; ++i) {
                    cand[i + 1u] = tail[i];
                }
                // peak = highest load this candidate would leave on any link it touches (multiplicity
                // included); tie-break on total hops over the group's mm-rings.
                touched.clear();
                uint32_t hops = 0;
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    for (uint32_t p = 0; p < 8u; ++p) {
                        const uint32_t a = cand[p], b = cand[(p + 1u) % 8u];
                        hops += gt[g].hp[mm][a][b];
                        for (const uint32_t l : gt[g].lk[mm][a][b]) {
                            touched.push_back(l);
                        }
                    }
                }
                std::sort(touched.begin(), touched.end());
                uint32_t peak = 0, i = 0;
                while (i < touched.size()) {
                    uint32_t j = i;
                    while (j < touched.size() && touched[j] == touched[i]) {
                        ++j;
                    }
                    peak = std::max(peak, load[touched[i]] + (j - i));
                    i = j;
                }
                if (peak < best_peak || (peak == best_peak && hops < best_hops)) {
                    best_peak = peak;
                    best_hops = hops;
                    best = cand;
                }
            } while (std::next_permutation(tail.begin(), tail.end()));
            chosen[g] = best;
            placed[g] = true;
            apply(g, best, +1);
        }
    }

    for (uint32_t g = 0; g < ngroups; ++g) {
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            const uint32_t jj = gt[g].base + mm;
            for (uint32_t pos = 0; pos < 8u; ++pos) {
                const uint32_t ci = chosen[g][pos] * preaders + jj;
                P.cores[ci].ring_pos = pos;
                P.cores[ci].ring_next_idx = chosen[g][(pos + 1u) % 8u] * preaders + jj;
                P.cores[ci].ring_prev_idx = chosen[g][(pos + 7u) % 8u] * preaders + jj;
            }
        }
    }
}

// TEST-ONLY (diag bit10 RING_BALANCED_BG): whole-op link-load-aware in0 ring ordering. Host-only,
// correctness-preserving; production ring MEMBERSHIP, only the visiting order changes.
//
// This is bit9 with the two corrections its measurements demanded (see IDEA1_RING_TOPOLOGY.md):
//
// (1) BACKGROUND TRAFFIC. bit9 balanced the ring against itself, which regressed 512x6144x4608 while
//     winning on 512x6144x2304 even though the two have IDENTICAL ring problems (same placement, same
//     128 KB shard, same chosen order) and differ only in in1 volume. Modelling all traffic shows why: on
//     4608 the busiest link is already at 10.52 MB of in1 read traffic and the ring adds only 0.16 MB, so
//     there was no headroom to win — the ring order was paying hops for nothing. So the in1 reads, the in0
//     own-shard read, the reduction chain and the output writes are all charged onto the link map FIRST
//     (they are fixed by the placement); the ring is then routed through that background's valleys.
//
// (2) LATENCY BUDGETS. bit9 minimized peak with only a total-hops tie-break, which cost -10% on
//     256x2048x2048 — a 24 KB shard, where the ring is LATENCY-bound (7 serial hops of a small payload) and
//     the worst directed edge sets the per-step time. Candidates are therefore constrained to never worsen
//     the group's worst edge (aggmax) and to stay within (1+kHopBudget) of production's total hops. Both
//     budgets are anchored on the order PRODUCTION would pick, so production's own order is always feasible
//     => the search can never be worse than production on either latency metric, only on link choice.
//
// Finally a GATE: the reordering is adopted only if it lowers the predicted peak by >= kMinGain; otherwise
// production's exact orders are kept. Combined, an unhelpful shape gets production behaviour rather than a
// regression.
void balance_in0_ring_order_bg(
    plan::ExecutionPlan& P,
    IDevice* device,
    const plan::Geometry& geo,
    uint32_t Pk,
    uint32_t Sm,
    const CoreCoord& grid) {
    namespace expd = tt::tt_metal::experimental::Device;
    constexpr double kHopBudget = 0.10;         // allow 10% more ring hops than production's order
    constexpr double kMinGain = 0.02;           // adopt only if the predicted peak drops by >= 2%
    constexpr uint64_t kLatencyShardBytes = 64u * 1024u;  // below this the ring is latency-, not bandwidth-bound
    const uint32_t preaders = geo.num_cores / 8u;
    const uint32_t ngroups = preaders / Sm;
    const RingLinkModel lm(device, grid.x, grid.y);

    // ---- Byte weights per traffic class (whole kernel, per core). ----
    const uint32_t kb_g = geo.K_slice_capacity / geo.K_num_blocks_eff;  // kb
    const uint64_t shard_bytes =
        static_cast<uint64_t>(geo.W) * geo.M_block_capacity * kb_g * kTileBytesBf16;  // W blocks of [M_block, kb]
    const uint64_t ring_edge_bytes = 7ull * shard_bytes;  // every ring edge carries 7 shards over the gather
    const uint64_t in1_bytes =
        static_cast<uint64_t>(geo.K_slice_capacity) * geo.N_sub * geo.N_bpc * kTileBytesBf16;
    const uint64_t red_bytes = static_cast<uint64_t>(geo.N_bpc) * geo.M_block_capacity * geo.N_sub * kTileBytesBf16;
    const uint64_t out_bytes = static_cast<uint64_t>(geo.M_block_capacity) * geo.N_sub * geo.N_bpc * kTileBytesBf16;

    std::vector<uint64_t> load(lm.num_links(), 0u);
    std::vector<uint32_t> ls;
    auto charge = [&](uint32_t sx, uint32_t sy, uint32_t tx, uint32_t ty, uint32_t noc, uint64_t b) {
        lm.links_rel(sx, sy, tx, ty, noc, ls);
        for (const uint32_t l : ls) {
            load[l] += b;
        }
    };

    // ---- Static background: everything the ring must share links with. ----
    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const plan::CorePlan& cp = P.cores[i];
        const uint32_t cx = lm.rx(cp.coord.x), cy = lm.ry(cp.coord.y);
        const uint32_t rnoc = cp.noc;                       // in1 reader NoC
        const uint32_t wnoc = (cp.noc == 0u) ? 1u : 0u;     // writer NoC (in0 ring / reduction / output)
        // in1: from this core's own bank (M-split slaves receive it from their mm==0 reader instead).
        if (Sm == 1u || cp.mm == 0u) {
            charge(lm.dram_rx(cp.bank), lm.ry(cp.coord.y), cx, cy, rnoc, in1_bytes);
            // NOTE: the DRAM endpoint row is the bank-adjacent worker's row; using this core's row is a
            // one-hop approximation only when the core was displaced by find_near.
        } else {
            const auto& rc = P.cores[i - cp.mm].coord;
            charge(lm.rx(rc.x), lm.ry(rc.y), cx, cy, rnoc, in1_bytes);
        }
        for (uint32_t b = 0; b < 8u; ++b) {  // in0 own shard: interleaved DRAM, spread over all banks
            charge(lm.dram_rx(b), lm.ry(P.cores[b * preaders].coord.y), cx, cy, wnoc, shard_bytes / 8u);
        }
        if (Pk > 1u && !cp.is_top) {
            const auto& nc = P.cores[cp.red_next_idx].coord;
            charge(cx, cy, lm.rx(nc.x), lm.ry(nc.y), wnoc, red_bytes);
        }
        if (Pk == 1u || cp.is_top) {  // output: core -> interleaved DRAM
            for (uint32_t b = 0; b < 8u; ++b) {
                charge(cx, cy, lm.dram_rx(b), lm.ry(P.cores[b * preaders].coord.y), wnoc, out_bytes / 8u);
            }
        }
    }
    const uint64_t bg_peak = *std::max_element(load.begin(), load.end());

    // ---- Per-group tables: edge -> links / hop cost, per mm-ring. ----
    struct GroupTables {
        uint32_t base{};
        NOC wnoc{};
        uint32_t wnoc_idx{};
        std::vector<std::array<std::array<std::vector<uint32_t>, 8>, 8>> lk;
        std::vector<std::array<std::array<uint32_t, 8>, 8>> hp;
    };
    std::vector<GroupTables> gt(ngroups);
    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t base = g * Sm;
        gt[g].base = base;
        gt[g].wnoc = (P.cores[base].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;
        gt[g].wnoc_idx = (gt[g].wnoc == NOC::NOC_0) ? 0u : 1u;
        gt[g].lk.resize(Sm);
        gt[g].hp.resize(Sm);
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t a = 0; a < 8u; ++a) {
                const auto& ca = P.cores[a * preaders + base + mm].coord;
                for (uint32_t b = 0; b < 8u; ++b) {
                    if (a == b) {
                        gt[g].hp[mm][a][b] = 0u;
                        continue;
                    }
                    const auto& cb = P.cores[b * preaders + base + mm].coord;
                    lm.links(ca, cb, gt[g].wnoc_idx, gt[g].lk[mm][a][b]);
                    gt[g].hp[mm][a][b] = expd::get_worker_noc_hop_distance(
                        device, CoreCoord{ca.x, ca.y}, CoreCoord{cb.x, cb.y}, gt[g].wnoc);
                }
            }
        }
    }
    auto agg = [&](uint32_t g, const std::array<uint32_t, 8>& ord) {
        uint32_t mx = 0, tot = 0;
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t p = 0; p < 8u; ++p) {
                const uint32_t e = gt[g].hp[mm][ord[p]][ord[(p + 1u) % 8u]];
                tot += e;
                mx = std::max(mx, e);
            }
        }
        return std::pair<uint32_t, uint32_t>{mx, tot};
    };
    auto apply = [&](uint32_t g, const std::array<uint32_t, 8>& ord, bool add) {
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            for (uint32_t p = 0; p < 8u; ++p) {
                for (const uint32_t l : gt[g].lk[mm][ord[p]][ord[(p + 1u) % 8u]]) {
                    load[l] = add ? (load[l] + ring_edge_bytes) : (load[l] - ring_edge_bytes);
                }
            }
        }
    };

    // ---- Production reference orders. This MUST reproduce optimize_in0_ring_order EXACTLY (the same
    // two-pass PARETO: pass 1 minimizes the mm==0 ring's (max, total) to establish the aggtot budget, pass 2
    // minimizes (aggmax, aggtot) subject to it) — otherwise the "keep production" fallback silently installs
    // a different order and the whole A/B is invalid. A single-pass min(aggmax, aggtot) is NOT the same for
    // Sm>1 and cost this experiment a spurious -6% on the Sm=3 shape.
    auto metrics4 = [&](uint32_t g, const std::array<uint32_t, 8>& ord) {
        struct M {
            uint32_t r0max, r0tot, aggmax, aggtot;
        } m{0, 0, 0, 0};
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            uint32_t mx = 0, tot = 0;
            for (uint32_t p = 0; p < 8u; ++p) {
                const uint32_t e = gt[g].hp[mm][ord[p]][ord[(p + 1u) % 8u]];
                tot += e;
                mx = std::max(mx, e);
            }
            if (mm == 0u) {
                m.r0max = mx;
                m.r0tot = tot;
            }
            m.aggmax = std::max(m.aggmax, mx);
            m.aggtot += tot;
        }
        return m;
    };
    auto lt2 = [](uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) { return a0 < b0 || (a0 == b0 && a1 < b1); };
    std::vector<std::array<uint32_t, 8>> prod(ngroups);
    for (uint32_t g = 0; g < ngroups; ++g) {
        const std::array<uint32_t, 8> bank = {0, 1, 2, 3, 4, 5, 6, 7};
        std::array<uint32_t, 8> opt_mm0 = bank;
        auto b_mm0 = metrics4(g, bank);
        b_mm0.r0max = ~0u;
        b_mm0.r0tot = ~0u;
        auto cand_of = [](const std::array<uint32_t, 7>& t) {
            std::array<uint32_t, 8> c{};
            c[0] = 0u;
            for (uint32_t i = 0; i < 7u; ++i) {
                c[i + 1u] = t[i];
            }
            return c;
        };
        std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
        do {
            const auto cand = cand_of(tail);
            const auto m = metrics4(g, cand);
            if (lt2(m.r0max, m.r0tot, b_mm0.r0max, b_mm0.r0tot)) {
                b_mm0 = m;
                opt_mm0 = cand;
            }
        } while (std::next_permutation(tail.begin(), tail.end()));
        std::array<uint32_t, 8> opt = opt_mm0;
        auto b_pa = b_mm0;
        const uint32_t budget = b_mm0.aggtot;
        std::array<uint32_t, 7> tail2 = {1, 2, 3, 4, 5, 6, 7};
        do {
            const auto cand = cand_of(tail2);
            const auto m = metrics4(g, cand);
            if (m.aggtot <= budget && lt2(m.aggmax, m.aggtot, b_pa.aggmax, b_pa.aggtot)) {
                b_pa = m;
                opt = cand;
            }
        } while (std::next_permutation(tail2.begin(), tail2.end()));
        prod[g] = opt;
        apply(g, opt, true);
    }
    const uint64_t prod_peak = *std::max_element(load.begin(), load.end());

    // ---- Balanced search under both latency budgets, two sequential passes. ----
    // Per-step ring time ~ max(worst_edge_hops * hop_latency, shard_bytes / link_bw). For a SMALL shard the
    // hop term dominates, so the worst edge must not grow (unbudgeted bit9 cost -10% on a 24 KB shard); for a
    // LARGE shard the bandwidth term dominates and that freedom is exactly what buys the win (+4.2% at 128 KB).
    const bool cap_edge = shard_bytes < kLatencyShardBytes;
    std::vector<std::array<uint32_t, 8>> chosen = prod;
    std::vector<uint32_t> touched;
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        for (uint32_t g = 0; g < ngroups; ++g) {
            apply(g, chosen[g], false);
            const auto pm = agg(g, prod[g]);
            const uint32_t max_hops = static_cast<uint32_t>(pm.second * (1.0 + kHopBudget));
            std::array<uint32_t, 8> best = prod[g];
            uint64_t best_peak = ~0ull;
            uint32_t best_hops = ~0u;
            std::array<uint32_t, 7> tail = {1, 2, 3, 4, 5, 6, 7};
            do {
                std::array<uint32_t, 8> cand{};
                cand[0] = 0u;
                for (uint32_t i = 0; i < 7u; ++i) {
                    cand[i + 1u] = tail[i];
                }
                const auto m = agg(g, cand);
                if ((cap_edge && m.first > pm.first) || m.second > max_hops) {
                    continue;  // (conditionally) never worsen the worst edge; always respect the hop budget
                }
                touched.clear();
                for (uint32_t mm = 0; mm < Sm; ++mm) {
                    for (uint32_t p = 0; p < 8u; ++p) {
                        for (const uint32_t l : gt[g].lk[mm][cand[p]][cand[(p + 1u) % 8u]]) {
                            touched.push_back(l);
                        }
                    }
                }
                std::sort(touched.begin(), touched.end());
                uint64_t peak = 0;
                for (size_t i = 0; i < touched.size();) {
                    size_t j = i;
                    while (j < touched.size() && touched[j] == touched[i]) {
                        ++j;
                    }
                    peak = std::max(peak, load[touched[i]] + static_cast<uint64_t>(j - i) * ring_edge_bytes);
                    i = j;
                }
                if (peak < best_peak || (peak == best_peak && m.second < best_hops)) {
                    best_peak = peak;
                    best_hops = m.second;
                    best = cand;
                }
            } while (std::next_permutation(tail.begin(), tail.end()));
            chosen[g] = best;
            apply(g, best, true);
        }
    }
    const uint64_t new_peak = *std::max_element(load.begin(), load.end());

    // ---- Gate: keep production unless the predicted peak improves materially. ----
    const bool adopt = static_cast<double>(new_peak) <= static_cast<double>(prod_peak) * (1.0 - kMinGain);
    const std::vector<std::array<uint32_t, 8>>& use = adopt ? chosen : prod;
    // log_info (not log_debug, which Release compiles out): this whole function only runs under the
    // diagnostic bit, so production stays silent, and the corpus A/B needs the adopt/keep decision to tell
    // "gate declined" (must be exactly neutral) from "gate adopted and gained nothing".
    log_info(
        tt::LogOp,
        "regime_a_matmul ring balance: background peak {} B, production peak {} B, balanced peak {} B -> {}",
        bg_peak,
        prod_peak,
        new_peak,
        adopt ? "ADOPT balanced" : "keep production");
    for (uint32_t g = 0; g < ngroups; ++g) {
        for (uint32_t mm = 0; mm < Sm; ++mm) {
            const uint32_t jj = gt[g].base + mm;
            for (uint32_t pos = 0; pos < 8u; ++pos) {
                const uint32_t ci = use[g][pos] * preaders + jj;
                P.cores[ci].ring_pos = pos;
                P.cores[ci].ring_next_idx = use[g][(pos + 1u) % 8u] * preaders + jj;
                P.cores[ci].ring_prev_idx = use[g][(pos + 7u) % 8u] * preaders + jj;
            }
        }
    }
}

// TEST-ONLY (diag bit12 PLACE_IN1_OPT): in1-read-optimal core PLACEMENT. Host-only and
// correctness-preserving - it only writes P.cores[i].coord, exactly like place_m_split_workers.
//
// Why the production placement is bad for in1: the read response travels DRAM endpoint -> core on the
// READER's NoC, dimension-ordered and strictly unidirectional with torus wrap (NOC_0 = +x/+y, NOC_1 =
// -x/-y). So each (bank, noc) has a cheap DOWNSTREAM region and everything else costs most of a lap.
// build_plan places every reader of a bank in one find_near spiral around opt[bank] - which is chosen "to the
// right of the DRAM controller", i.e. downstream for NOC_0 only. Worse, IDevice::
// get_optimal_dram_bank_to_logical_worker_assignment CACHES its result without keying on the NoC
// (device.cpp: `if (optimal_dram_bank_to_logical_worker_assignment_.empty())`), so opt1 == opt0 in practice
// and the noc==1 readers are placed at the NOC_0-optimal core. Measured consequence: NOC_0 readers average
// 9.0 hops, NOC_1 readers 17.2, and NOC_1 carries 66% of in1 read hop-bytes with half the readers.
//
// This installs the CROSS layout instead: each (bank, noc) group is placed in the region downstream of THAT
// endpoint on THAT NoC (nearest-first), so a bank's NOC_0 readers sit on one side of its DRAM column and its
// NOC_1 readers on the other. Offline (in1_place_search.py, exact route model): in1 read hops -69..-78%,
// peak in1 link load exactly at the endpoint-egress floor (-50..-57%), whole-op peak -36..-50%, and the in0
// ring gets 13-23% cheaper as a side effect. An exact min-cost assignment (Hungarian) over the same cost
// matrix beats this deterministic heuristic by only 0-2% on in1 hops and is WORSE on whole-op peak, so the
// heuristic is what is implemented.
void place_in1_optimal(plan::ExecutionPlan& P, IDevice* device, const plan::Geometry& geo, const CoreCoord& grid) {
    namespace expd = tt::tt_metal::experimental::Device;
    const uint32_t preaders = geo.num_cores / 8u;
    const RingLinkModel lm(device, grid.x, grid.y);
    const auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);

    // Per-(bank, noc) DRAM endpoint PHYSICAL row, from blackhole_140_arch.yaml `dram` x `dram_views.
    // worker_endpoint` (subchannel per NoC). BH-specific static data because no IDevice accessor exposes the
    // per-NoC endpoint (dram_core_from_dram_channel is on Device, not IDevice) - on promotion this must come
    // from an API. The NOC_0 column is ASSERTED against the device below, which validates the indexing.
    constexpr uint32_t kEpRow[8][2] = {{11, 1}, {2, 10}, {9, 4}, {5, 7}, {11, 1}, {3, 10}, {8, 4}, {6, 7}};
    constexpr uint32_t kPhysYOff = 2u;  // physical y of logical row 0
    for (uint32_t b = 0; b < 8u; ++b) {
        TT_FATAL(
            kEpRow[b][0] == opt0[b].y + kPhysYOff,
            "regime_a_matmul in1 placement: DRAM endpoint table is stale for this device (bank {} expects "
            "physical row {}, device's NOC_0-optimal worker is at logical row {})",
            b,
            kEpRow[b][0],
            opt0[b].y);
    }
    // Endpoint position in the model's relative frame. x: the DRAM column, derived from which side the
    // bank-adjacent worker sits on (left column => one step upstream of logical x=0; right column => inside
    // the middle gap). y: the table above.
    auto ep_rel = [&](uint32_t b, uint32_t noc) {
        const uint32_t rx = lm.dram_rx(b);
        const uint32_t ry = lm.ry(0) + kEpRow[b][noc] - kPhysYOff;  // rel y == logical row for BH
        return std::pair<uint32_t, uint32_t>{rx, ry % 12u ? (ry % 12u) : 0u};
    };

    // ---- CROSS placement of the DRAM readers (mm == 0) ----
    std::set<std::pair<uint32_t, uint32_t>> used;
    std::vector<uint32_t> links;
    auto resp_hops = [&](uint32_t b, uint32_t noc, uint32_t x, uint32_t y) {
        const auto e = ep_rel(b, noc);
        lm.links_rel(e.first, e.second, lm.rx(x), lm.ry(y), noc, links);
        return static_cast<uint32_t>(links.size());
    };
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t noc = 0; noc < 2u; ++noc) {
            // slots of this (bank, noc): the mm==0 cores whose reader NoC is `noc`
            std::vector<uint32_t> slots;
            for (uint32_t p = 0; p < preaders; ++p) {
                const uint32_t i = b * preaders + p;
                if (P.cores[i].mm == 0u && P.cores[i].noc == noc) {
                    slots.push_back(i);
                }
            }
            if (slots.empty()) {
                continue;
            }
            // candidate cells, nearest-first in RESPONSE distance on this NoC (ties by y then x => rows fill)
            std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> cand;
            for (uint32_t y = 0; y < grid.y; ++y) {
                for (uint32_t x = 0; x < grid.x; ++x) {
                    cand.push_back({resp_hops(b, noc, x, y), {x, y}});
                }
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& c) {
                if (a.first != c.first) {
                    return a.first < c.first;
                }
                return a.second.second != c.second.second ? a.second.second < c.second.second
                                                          : a.second.first < c.second.first;
            });
            size_t ci = 0;
            for (const uint32_t i : slots) {
                while (ci < cand.size() && used.count(cand[ci].second)) {
                    ++ci;
                }
                if (ci >= cand.size()) {
                    break;
                }
                used.insert(cand[ci].second);
                P.cores[i].coord.x = cand[ci].second.first;
                P.cores[i].coord.y = cand[ci].second.second;
                ++ci;
            }
        }
    }

    // ---- M-split slaves: IN1_NEAR, same rule as place_m_split_workers (they never read DRAM) ----
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (P.cores[i].mm == 0u) {
                continue;
            }
            const CoreCoord rc{P.cores[i - P.cores[i].mm].coord.x, P.cores[i - P.cores[i].mm].coord.y};
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
            P.cores[i].coord.x = best.x;
            P.cores[i].coord.y = best.y;
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
void place_mesh(plan::ExecutionPlan& P, const plan::Geometry& geo, const CoreCoord& grid, bool spread_rows) {
    const uint32_t preaders = geo.num_cores / 8u;
    const uint32_t overflow_cols = (grid.x > 8u) ? (grid.x - 8u) : 0u;
    TT_FATAL(
        grid.x >= 8u && preaders <= grid.y + overflow_cols,
        "regime_a_matmul mesh placement does not fit: {} slices need <= {} (grid {}x{})",
        preaders,
        grid.y + overflow_cols,
        grid.x,
        grid.y);
    for (uint32_t b = 0; b < 8u; ++b) {
        for (uint32_t p = 0; p < preaders; ++p) {
            const uint32_t i = b * preaders + p;
            if (p < grid.y) {
                P.cores[i].coord.x = b;  // banks along x, slices along y
                // spread_rows: when there are fewer slices than rows, SPACE THEM OUT over all the rows instead
                // of packing rows 0..preaders-1. Packing is what makes the mesh catastrophic at small
                // preaders (-48% to -89%): every core ends up in one corner and all the DRAM paths pile onto
                // the same links. Identity when preaders >= grid.y, so the shipped gate is unaffected.
                P.cores[i].coord.y = (spread_rows && preaders <= grid.y) ? (p * grid.y) / preaders : p;
            } else {
                P.cores[i].coord.x = 8u + (p - grid.y);  // one spare column per overflow slice
                P.cores[i].coord.y = b;
            }
        }
    }
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

    // ---- Kernel compile defines. wdefs = writer (in0 ring/reduce + fused output); fdefs_compute (below) =
    // compute fusion defines merged into cdefs. The in1 reader takes NO defines. Empty maps => the
    // byte-identical no-fusion compile. ----
    std::map<std::string, std::string> wdefs;

    // ---- TEST-ONLY critical-path ablation (operation_attributes.diag_mask; 0 => production, no define, no
    // extra arg => byte-identical). 6 combinable bits -> writer/compute kernel defines. Only the unfused,
    // single-output path is supported (the read-skip flag is appended at writer arg index 17, which must be
    // free). Compute defines (SKIP_COMPUTE / SKIP_REDUCTION) are merged into cdefs at compute-kernel creation. ----
    const uint32_t diag_mask = operation_attributes.diag_mask;
    std::map<std::string, std::string> ddefs_compute;  // diagnostic compute defines
    // Bits 0..7 and bit11 alter kernel behaviour (and produce invalid output) => restricted to
    // unfused/single-output. Bits 8..10 are host-only + correctness-preserving, so they are allowed everywhere.
    constexpr uint32_t kDiagKernelBits = 0xFFu | 0x800u;
    if ((diag_mask & kDiagKernelBits) != 0u) {
        TT_FATAL(
            !has_bias && !has_ternary && !has_activation && n_chunks == 1u,
            "regime_a_matmul ablation diagnostic (diag_mask={}) is only supported unfused + single-output",
            diag_mask);
        if (diag_mask & 0x1u) {
            wdefs["SKIP_ALL_IN0_READ"] = "1";
        }
        if (diag_mask & 0x2u) {
            wdefs["SKIP_REDUNDANT_IN0_READ"] = "1";
        }
        if (diag_mask & 0x4u) {
            wdefs["SKIP_IN0_RING_FORWARD"] = "1";
        }
        if (diag_mask & 0x8u) {
            ddefs_compute["SKIP_COMPUTE"] = "1";
        }
        if (diag_mask & 0x10u) {
            wdefs["SKIP_REDUCTION"] = "1";
            ddefs_compute["SKIP_REDUCTION"] = "1";
        }
        if (diag_mask & 0x20u) {
            wdefs["SKIP_OUTPUT_WRITE"] = "1";
        }
        if (diag_mask & 0x40u) {
            wdefs["FWD_NEAR"] = "1";
        }
        if (diag_mask & 0x80u) {
            wdefs["FWD_HALF"] = "1";
        }
    }
    const bool diag_read_skip_arg = (diag_mask & 0x3u) != 0u;  // bit0/bit1 => per-core read-skip flag at arg 17
    const bool diag_near_arg = (diag_mask & 0x40u) != 0u;      // bit6 => per-core nearest-peer coords follow

    // ---- M-split worker PLACEMENT (Sm>1): IN1_NEAR. Overrides only P.cores[i].coord; MUST run BEFORE the ring
    // reorder so the ring order recomputes on the placed coords. No-op at Sm==1. ----
    const bool diag_place_in1 = (diag_mask & 0x1000u) != 0u;   // bit12: in1-read-optimal placement (diag)
    const bool diag_place_mesh = (diag_mask & 0x2000u) != 0u;  // bit13: force the mesh on
    const bool diag_mesh_off = (diag_mask & 0x4000u) != 0u;    // bit14: force the mesh off (A/B the default)
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
    const bool mesh_gate = ((Pk * Ns_gate >= 10u) && (Sm == 1u) && (Ns_gate == 1u || Pk >= 4u)) ||
                           (ring_bytes >= 2u * in1_bytes);
    const bool diag_mesh_spread = (diag_mask & 0x8000u) != 0u;  // bit15: even row spacing for few slices
    if ((diag_place_mesh || mesh_gate) && !diag_mesh_off) {
        place_mesh(P, geo, device->compute_with_storage_grid_size(), diag_mesh_spread);
    } else if (diag_place_in1) {
        place_in1_optimal(P, device, geo, device->compute_with_storage_grid_size());
    } else if (Sm > 1u) {
        place_m_split_workers(P, device, geo);
    }

    // ---- Physical-topology-aware in0 ring ordering (PARETO) over each (kk,nn) group's Sm mm-rings. ----
    // diag bit8 (RING_REGIONAL) instead re-partitions the rings across nn for physical compactness (host-only,
    // correctness-preserving); it subsumes the ordering step. mask 0 keeps the production path byte-identical.
    const uint32_t Ns_cfg = cfg.n_slices ? cfg.n_slices : 1u;
    if ((diag_mask & 0x100u) != 0u && Ns_cfg > 1u) {
        regroup_in0_rings(P, device, geo, Pk, Ns_cfg, Sm);
    } else if ((diag_mask & 0x400u) != 0u) {
        balance_in0_ring_order_bg(P, device, geo, Pk, Sm, device->compute_with_storage_grid_size());
    } else if ((diag_mask & 0x200u) != 0u) {
        balance_in0_ring_order(P, device, geo, Sm, device->compute_with_storage_grid_size());
    } else {
        optimize_in0_ring_order(P, device, geo, Sm);
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
    // in1 reader defines: none in production; bit11 compile-gates out its DRAM read payload.
    std::map<std::string, std::string> rdefs;
    if (diag_mask & 0x800u) {
        rdefs["SKIP_IN1_READ"] = "1";
    }
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
    cdefs.insert(fdefs_compute.begin(), fdefs_compute.end());  // fusion defines (empty for the no-fusion path)
    cdefs.insert(ddefs_compute.begin(), ddefs_compute.end());  // diagnostic defines (empty for mask 0)
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

    // TEST-ONLY (bit6 FWD_NEAR): per-core NEAREST other program core on that core's WRITER NoC, used as a
    // stand-in ring-forward payload destination. Attributes the ring-forward cost between hop distance /
    // link contention (this perturbation removes distance but not bytes) and per-core injection bandwidth.
    // Only computed for the diagnostic build; production (mask 0) never enters here.
    std::vector<CoreCoord> diag_near(diag_near_arg ? geo.num_cores : 0u);
    if (diag_near_arg) {
        namespace expd = tt::tt_metal::experimental::Device;
        for (uint32_t i = 0; i < geo.num_cores; ++i) {
            const CoreCoord src{P.cores[i].coord.x, P.cores[i].coord.y};
            const NOC wnoc = (P.cores[i].noc == 0u) ? NOC::NOC_1 : NOC::NOC_0;  // writer NoC (opposite reader)
            uint32_t bestd = 0xffffffffu;
            CoreCoord best = src;
            for (uint32_t j = 0; j < geo.num_cores; ++j) {
                if (j == i) {
                    continue;
                }
                const CoreCoord dst{P.cores[j].coord.x, P.cores[j].coord.y};
                const uint32_t d = expd::get_worker_noc_hop_distance(device, src, dst, wnoc);
                if (d < bestd) {
                    bestd = d;
                    best = dst;
                }
            }
            diag_near[i] = device->worker_core_from_logical_core(best);
        }
    }

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
        // TEST-ONLY in0-read-skip flag (appended at writer arg index 17 for the unfused/single-output
        // diagnostic build; absent => production arg layout unchanged). 1 => this core skips its in0 DRAM
        // read. bit0 (SKIP_ALL) dominates bit1 (SKIP_REDUNDANT, ns>0 only) => normalizes skip-all+redundant.
        if (diag_read_skip_arg) {
            const uint32_t skip = (diag_mask & 0x1u) ? 1u : (cp.nn > 0u ? 1u : 0u);
            wa.push_back(skip);
        }
        // TEST-ONLY (bit6): nearest-peer physical coords, appended AFTER the optional read-skip flag.
        if (diag_near_arg) {
            wa.push_back(diag_near[i].x);
            wa.push_back(diag_near[i].y);
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
