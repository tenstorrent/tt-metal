// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_tail_plan.hpp"

#include <algorithm>
#include <numeric>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

namespace fused_tail_plan_detail {

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// len(range(start, stop, step))
uint32_t range_len(uint32_t start, uint32_t stop, uint32_t step) {
    return start < stop ? ceil_div(stop - start, step) : 0;
}

}  // namespace fused_tail_plan_detail
using namespace fused_tail_plan_detail;

std::optional<uint32_t> row_neighbor(uint32_t p, uint32_t G, bool ring, TailDir d) {
    if (d == TailDir::Fwd) {
        return (ring || p + 1 < G) ? std::optional<uint32_t>((p + 1) % G) : std::nullopt;
    }
    return (ring || p > 0) ? std::optional<uint32_t>((p + G - 1) % G) : std::nullopt;
}

std::pair<std::vector<Unit>, std::vector<Unit>> gather_schedule(uint32_t p, uint32_t G, bool ring) {
    std::vector<Unit> fwd, bwd;
    if (ring && G % 2 == 0 && G >= 4) {
        const uint32_t h = G / 2;
        for (uint32_t i = 0; i < h; ++i) {
            fwd.push_back({(p + G - i) % G, i == h - 1 ? 1u : 0u});
            bwd.push_back({(p + i) % G, i == h - 1 ? 2u : 0u});
        }
        return {fwd, bwd};
    }
    if (ring) {
        const uint32_t kf = G / 2, kb = G - 1 - kf;
        for (uint32_t i = 0; i < kf; ++i) {
            fwd.push_back({(p + G - i) % G, 0});
        }
        for (uint32_t i = 0; i < kb; ++i) {
            bwd.push_back({(p + i) % G, 0});
        }
        return {fwd, bwd};
    }
    if (p + 1 < G) {
        for (uint32_t i = 0; i <= p; ++i) {
            fwd.push_back({p - i, 0});
        }
    }
    if (p > 0) {
        for (uint32_t i = 0; i < G - p; ++i) {
            bwd.push_back({p + i, 0});
        }
    }
    return {fwd, bwd};
}

std::vector<uint32_t> chunk_order(uint32_t p, uint32_t G, bool ring) {
    auto dist = [&](uint32_t q) { return ring ? std::min((q + G - p) % G, (p + G - q) % G) : (q > p ? q - p : p - q); };
    std::vector<uint32_t> out(G);
    std::iota(out.begin(), out.end(), 0u);
    std::stable_sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) {
        const auto key = [&](uint32_t q) { return std::make_tuple(dist(q), (q + G - p) % G != G - 1, q); };
        return key(a) < key(b);
    });
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> even_split(uint32_t n, uint32_t parts) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    const uint32_t base = n / parts, rem = n % parts;
    uint32_t k = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t w = base + (i < rem ? 1 : 0);
        out.emplace_back(k, k + w);
        k += w;
    }
    return out;
}

std::vector<uint32_t> column_split(uint32_t n_tiles, uint32_t columns, uint32_t unit) {
    TT_FATAL(n_tiles % unit == 0, "fused_tail: {} tiles in units of {}", n_tiles, unit);
    const uint32_t base = n_tiles / unit / columns, rem = n_tiles / unit % columns;
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < columns; ++i) {
        out.push_back((base + (i < rem ? 1 : 0)) * unit);
    }
    return out;
}

uint32_t ring_pages(uint32_t pages, const std::vector<uint32_t>& pushes) {
    const std::set<uint32_t> ps(pushes.begin(), pushes.end());
    if (ps.size() == 1 && pages % *ps.begin() == 0) {
        return pages;
    }
    return pages + *ps.rbegin() - 1;
}

// ---- TailSchedule ----

TailSchedule::TailSchedule(
    uint32_t G,
    bool ring,
    uint32_t L,
    uint32_t NTB,
    uint32_t RT,
    uint32_t NBK,
    uint32_t RP,
    std::map<std::pair<uint32_t, uint32_t>, TailDir> ties) :
    G(G), L(L), NTB(NTB), RT(RT), NBK(NBK), RP(RP), ring(ring), ties(std::move(ties)) {
    for (uint32_t p = 0; p < G; ++p) {
        for (TailDir d : {TailDir::Fwd, TailDir::Bwd}) {
            for (uint32_t l = 0; l < L; ++l) {
                for (const auto& [q, part] : dests(p, d)) {
                    // (my arrivals: the destination's port facing the side my packets come in by; my ready: its port
                    // facing the fabric's way back to me)
                    expect_[{q, opposite(d), l}] += ceil_div(RT * runs(part, l), kIncEvery);
                    readies_[{q, dir(q, p), l}].insert({p, d, l});
                }
            }
        }
    }
}

TailDir TailSchedule::dir(uint32_t p, uint32_t q) const {
    if (!ring) {
        return q > p ? TailDir::Fwd : TailDir::Bwd;
    }
    const uint32_t df = (q + G - p) % G;
    if (2 * df == G) {
        auto it = ties.find({p, q});
        return it == ties.end() ? TailDir::Fwd : it->second;
    }
    return 2 * df < G ? TailDir::Fwd : TailDir::Bwd;
}

std::vector<Unit> TailSchedule::dests(uint32_t p, TailDir d) const {
    std::vector<Unit> out;
    if (!ring) {
        if (d == TailDir::Fwd) {
            for (uint32_t q = p + 1; q < G; ++q) {
                out.push_back({q, 0});
            }
        } else {
            for (uint32_t q = p; q-- > 0;) {
                out.push_back({q, 0});
            }
        }
        return out;
    }
    for (uint32_t k = 1; k < G; ++k) {
        const uint32_t q = d == TailDir::Fwd ? (p + k) % G : (p + G - k) % G;
        if (dir(p, q) == d) {
            out.push_back({q, 0});
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> TailSchedule::run_list(uint32_t part, uint32_t l) const {
    const uint32_t nt = std::min(NBK, NTB);
    const uint32_t nl = range_len(l, nt, L);
    const uint32_t lo = part == 2 ? nl / 2 : 0, hi = part == 1 ? nl / 2 : nl;
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t i = lo; i < hi; ++i) {
        const uint32_t t0 = l + i * L;
        for (uint32_t m = t0; m < NTB; m += RP * NBK) {
            out.emplace_back(m, std::min(RP, ceil_div(NTB - m, NBK)));
        }
    }
    return out;
}

uint32_t TailSchedule::hops(uint32_t p, uint32_t q, TailDir d) const {
    return d == TailDir::Fwd ? (q + G - p) % G : (p + G - q) % G;
}

uint32_t TailSchedule::expect(const Port& port) const {
    auto it = expect_.find(port);
    return it == expect_.end() ? 0 : it->second;
}

std::vector<TailSchedule::Port> TailSchedule::readies(const Port& port) const {
    auto it = readies_.find(port);
    if (it == readies_.end()) {
        return {};
    }
    std::vector<Port> out(it->second.begin(), it->second.end());
    // (the reference sorts (p, "fwd" / "bwd", l): "bwd" first)
    std::sort(out.begin(), out.end(), [](const Port& a, const Port& b) {
        return std::make_tuple(std::get<0>(a), std::get<1>(a) == TailDir::Fwd, std::get<2>(a)) <
               std::make_tuple(std::get<0>(b), std::get<1>(b) == TailDir::Fwd, std::get<2>(b));
    });
    return out;
}

TailA2AGeometry TailA2AGeometry::make(uint32_t G, uint32_t rows, uint32_t H, uint32_t NBK, size_t max_payload_bytes) {
    TailA2AGeometry g;
    g.G = G;
    g.rows = rows;
    g.H = H;
    g.S = rows / G;
    TT_FATAL(
        rows % G == 0 && g.S % 32 == 0 && H % (32 * G) == 0,
        "fused_tail (a2a): {} rows, H {} over TP {}: whole tiles",
        rows,
        H,
        G);
    g.RT = g.S / 32;
    g.NTH = H / 32;
    g.NTB = g.NTH / G;
    for (uint32_t d = 1; d <= 8; ++d) {
        if (g.NTB % d == 0) {
            g.W = d;
        }
    }
    g.NW = g.NTH / g.W;
    g.NBK = NBK;
    g.RP = std::max<uint32_t>(1, static_cast<uint32_t>(max_payload_bytes / 2048));
    g.group = std::max<uint32_t>(1, std::min<uint32_t>(8, (96 * 1024) / (2 * g.RP * 2048)));
    return g;
}

std::shared_ptr<const TailA2APlan> make_tail_a2a_plan(MoEBlockResources& resources, uint32_t rows, uint32_t H) {
    auto& mesh = resources.mesh_device();
    const auto& topo = resources.topology();
    const uint32_t G = topo.C;
    TT_FATAL(G >= 2, "fused_tail (a2a): a TP group (mesh row) of {} chips", G);
    const auto dram = mesh.dram_grid_size();
    auto plan = std::make_shared<TailA2APlan>();
    plan->geom = TailA2AGeometry::make(G, rows, H, dram.x * dram.y, resources.fabric_max_payload_bytes());
    plan->ring = topo.wrap_x;
    plan->num_links = resources.config().num_links;
    const auto& g = plan->geom;

    auto node = [&](ChipCoord ch) {
        return mesh.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(ch.r, ch.c));
    };
    // The ring's far side (distance G / 2): the way the fabric routes it from p, the same on every mesh row.
    std::map<std::pair<uint32_t, uint32_t>, TailDir> ties;
    if (plan->ring && G % 2 == 0) {
        for (uint32_t r = 0; r < topo.R; ++r) {
            for (uint32_t p = 0; p < G; ++p) {
                const uint32_t q = (p + G / 2) % G;
                const auto fd = tt::tt_fabric::get_eth_forwarding_direction(node({r, p}), node({r, (p + 1) % G}));
                const auto qd = tt::tt_fabric::get_eth_forwarding_direction(node({r, p}), node({r, q}));
                const TailDir d = qd == fd ? TailDir::Fwd : TailDir::Bwd;
                const auto [it, fresh] = ties.emplace(std::make_pair(p, q), d);
                TT_FATAL(fresh || it->second == d, "fused_tail (a2a): ring ties differ between mesh rows");
            }
        }
    }
    plan->links = row_links(resources.mesh_plan());
    plan->sched = TailSchedule(G, plan->ring, plan->num_links, g.NTB, g.RT, g.NBK, g.RP, std::move(ties));

    // Ports: one per (direction, link) with a neighbour on the row, next to its link's Ethernet core (NoC1).
    MeshPlan row_plan = resources.mesh_plan();
    for (auto& [chip, dirs] : row_plan.chips) {
        dirs[static_cast<size_t>(Dir::YPlus)].reset();
        dirs[static_cast<size_t>(Dir::YMinus)].reset();
    }
    const auto cores = place_ports(mesh, row_plan);
    std::set<std::pair<size_t, size_t>> taken;
    std::vector<CoreRange> ranges;
    for (const auto& [key, core] : cores) {
        const TailDir d = key.dir == Dir::XPlus ? TailDir::Fwd : TailDir::Bwd;
        plan->ports[{key.chip, d, key.link_index}] = core;
        if (taken.insert({core.y, core.x}).second) {
            ranges.emplace_back(core, core);
        }
    }
    plan->port_union = CoreRangeSet(ranges);
    const auto grid = mesh.compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y && plan->workers.size() < g.NW; ++y) {
        for (uint32_t x = 0; x < grid.x && plan->workers.size() < g.NW; ++x) {
            if (!taken.contains({y, x})) {
                plan->workers.emplace_back(x, y);
            }
        }
    }
    TT_FATAL(plan->workers.size() == g.NW, "fused_tail (a2a): {} workers, {} free cores", g.NW, plan->workers.size());
    return plan;
}

// ---- TailEngine ----

TailEngineGeometry TailEngineGeometry::make(
    uint32_t rows,
    uint32_t Hr,
    uint32_t Ho,
    uint32_t chunks,
    uint32_t GX,
    uint32_t GY,
    uint32_t wtb,
    uint32_t cap_bytes) {
    TailEngineGeometry g;
    TT_FATAL(GY % 2 == 0, "fused_tail (engine): GY {} is odd", GY);
    TT_FATAL(Hr % 64 == 0 && Ho % 32 == 0, "fused_tail (engine): Hr {} / Ho {}: whole tiles of each K half", Hr, Ho);
    TT_FATAL(
        rows % (32 * chunks) == 0, "fused_tail (engine): {} rows: whole row tiles a chunk ({} chunks)", rows, chunks);
    g.rows = rows;
    g.Hr = Hr;
    g.Ho = Ho;
    g.chunks = chunks;
    g.GX = GX;
    g.GY = GY;
    g.wtb = wtb;
    g.KT = Hr / 32;
    g.NTO = Ho / 32;
    g.RT = rows / 32 / chunks;
    g.PAIRS = GY / 2;
    const uint32_t NT = chunks * g.RT;
    for (uint32_t p = 0; p < g.PAIRS; ++p) {
        g.nst.push_back(range_len(p, NT, g.PAIRS));
        g.nown.push_back(range_len(p, g.RT, g.PAIRS));
    }
    g.NSTMAX = *std::max_element(g.nst.begin(), g.nst.end());
    g.NOWNMAX = *std::max_element(g.nown.begin(), g.nown.end());
    g.kh = even_split(g.KT, 2);
    g.KH = g.kh[0].second;
    TT_FATAL(g.kh[1].second - g.kh[1].first == g.KH, "fused_tail (engine): equal K halves");
    g.pn = column_split(g.NTO, GX, 1);
    uint32_t acc = 0;
    for (uint32_t cx = 0; cx < GX; ++cx) {
        g.n0.push_back(acc);
        acc += g.pn[cx];
    }
    g.PNMAX = *std::max_element(g.pn.begin(), g.pn.end());
    TT_FATAL(g.PNMAX <= 7, "fused_tail (engine): a step's {} fp32 tiles in DEST", g.PNMAX);
    // (the columns a core finishes: half 0 the first ceil(pn / 2), half 1 the rest - at most XT - 1 either way)
    std::set<uint32_t> fns;
    for (uint32_t p : g.pn) {
        const uint32_t c = ceil_div(p, 2);
        g.fin.push_back({std::make_pair(0u, c), std::make_pair(c, p - c)});
        for (const auto& [fo, fn] : g.fin.back()) {
            TT_FATAL(fn > 0 && fn <= kXT - 1 && p - fn <= kXT - 1, "fused_tail (engine): {} columns a core", p);
            fns.insert(fn);
        }
    }
    g.FNS.assign(fns.begin(), fns.end());
    uint32_t lcm = 1;
    for (uint32_t f : g.FNS) {
        lcm = std::lcm(lcm, f);
    }
    g.FNR = lcm * ceil_div(2 * g.FNS.back(), lcm);  // (the finished columns' FIFOs: a multiple of every core's)
    g.wk = even_split(g.KH, g.PAIRS);
    g.ak = even_split(g.KH, GX);
    for (const auto& [k0, k1] : g.ak) {
        g.SHMAX = std::max(g.SHMAX, k1 - k0);
    }
    // the z ring and the E slots: a slot a step where L1 has room (no slot reused: no FREE waits), else fewer
    const uint32_t lo = std::min<uint32_t>(2, g.NSTMAX);
    uint32_t nslot = g.NSTMAX, total = 0;
    for (;; --nslot) {
        total = g.layout(nslot).second;
        if (total <= cap_bytes || nslot == lo) {
            break;
        }
    }
    TT_FATAL(
        total <= cap_bytes,
        "fused_tail (engine): its L1 {} KB > {} KB at {} ring slots",
        total >> 10,
        cap_bytes >> 10,
        lo);
    g.NSLOT = nslot;
    g.arena_bytes = total;
    return g;
}

std::pair<std::vector<uint32_t>, uint32_t> TailEngineGeometry::layout(uint32_t nslot) const {
    auto al = [](uint32_t v) { return (v + 63) / 64 * 64; };
    const std::vector<uint32_t> sizes = {
        KH * PNMAX * wtb,        // w
        nslot * KH * 2048,       // ring
        NSTMAX * SHMAX * 2048,   // land
        nslot * GX * 4096,       // part
        4096,                    // pmy
        2048,                    // ones
        FNR * 4096,              // accm
        2 * kXT * 4096,          // xsend
        2 * 4096,                // sqm
        NSTMAX * kXT * 4096,     // xrecv
        FNR * 2048,              // se
        FNR * 2048,              // out
        NOWNMAX * SHMAX * 2048,  // rm
    };
    std::vector<uint32_t> off;
    uint32_t o = 0;
    for (uint32_t v : sizes) {
        off.push_back(o);
        o = al(o + v);
    }
    return {off, o};
}

uint32_t TailEngineGeometry::region_tiles() const {
    uint32_t m = 0;
    for (const auto& [k0, k1] : wk) {
        m = std::max(m, k1 - k0);
    }
    return m * PNMAX;
}

std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> engine_relays(
    uint32_t p, uint32_t G, bool ring, const TailEngineGeometry& g, uint32_t wg) {
    std::set<Unit> units;  // (every block landed here by a port: from either neighbour)
    for (uint32_t c = 0; c < G; ++c) {
        const auto [fwd, bwd] = gather_schedule(c, G, ring);
        if (row_neighbor(c, G, ring, TailDir::Fwd) == p) {
            units.insert(fwd.begin(), fwd.end());
        }
        if (row_neighbor(c, G, ring, TailDir::Bwd) == p) {
            units.insert(bwd.begin(), bwd.end());
        }
    }
    const auto order = chunk_order(p, g.chunks, ring);
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> out;
    for (const auto& [q, part] : units) {
        const uint32_t c0 = part == 2 ? g.GX / 2 : 0, c1 = part == 1 ? g.GX / 2 : g.GX;
        const uint32_t pos = static_cast<uint32_t>(std::find(order.begin(), order.end(), q) - order.begin());
        for (uint32_t r = 0; r < g.RT; ++r) {
            const uint32_t t = pos * g.RT + r, pr = t % g.PAIRS, s = t / g.PAIRS;
            for (uint32_t g0 = c0; g0 < c1; g0 += wg) {
                const uint32_t g1 = std::min(g0 + wg, c1);
                for (uint32_t h = 0; h < 2; ++h) {
                    if (g1 - g0 > 1) {
                        out[{g1 - 1, pr + g.PAIRS * h}].push_back(s | g0 << 8 | g1 << 16);
                    }
                }
            }
        }
    }
    for (auto& [k, v] : out) {
        TT_FATAL(v.size() <= std::min<uint32_t>(g.NSTMAX, 32), "fused_tail (engine): {} watched steps", v.size());
        std::sort(v.begin(), v.end());
    }
    return out;
}

std::shared_ptr<const TailEnginePlan> make_tail_engine_plan(
    MoEBlockResources& resources, uint32_t rows, uint32_t Hr, uint32_t Ho, uint32_t wtb, uint32_t cap_bytes) {
    auto& mesh = resources.mesh_device();
    const auto& topo = resources.topology();
    const auto grid = mesh.compute_with_storage_grid_size();
    // (the engine: every row of the grid, an even count of them, and every column but the last)
    const uint32_t GY = grid.y / 2 * 2;
    TT_FATAL(grid.x >= 2 && GY >= 4, "fused_tail (engine): a {}x{} grid (at least 2 columns, 4 rows)", grid.x, grid.y);
    TT_FATAL(
        resources.config().num_links == 2,
        "fused_tail (engine): {} links (a link a K half)",
        resources.config().num_links);
    const uint32_t GX = grid.x - 1;  // (the last column: the gather's ports)
    auto plan = std::make_shared<TailEnginePlan>();
    plan->geom = TailEngineGeometry::make(rows, Hr, Ho, topo.C, GX, GY, wtb, cap_bytes);
    plan->ring = topo.wrap_x;
    plan->num_links = 2;
    plan->nbank = mesh.dram_grid_size().x;
    plan->RP = std::max<uint32_t>(1, static_cast<uint32_t>(resources.fabric_max_payload_bytes() / 2048));
    plan->norm_eps = resources.dims().norm_eps;
    plan->links = row_links(resources.mesh_plan());
    // (the ports' column has no row with room for a placement search: placed down it - fwd above, bwd below, a link
    // apart)
    const std::array<uint32_t, 4> port_rows = {GY / 10, 3 * GY / 10, 6 * GY / 10, 8 * GY / 10};
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t l = 0; l < 2; ++l) {
            const CoreCoord core(GX, port_rows[2 * i + l]);
            plan->ports[{i == 0 ? TailDir::Fwd : TailDir::Bwd, l}] = core;
            ranges.emplace_back(core, core);
        }
    }
    plan->port_cores = CoreRangeSet(ranges);
    plan->engine_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(GX - 1, GY - 1)));
    const auto& g = plan->geom;
    // (a port core holds the landing CB (its base names the engine's landing slots), its staging (three groups: one
    // sending, one landing, one being issued), the group counts and the poll words)
    const uint32_t port_bytes =
        g.NSTMAX * g.SHMAX * 2048 + 3 * kPortGroup * plan->RP * 2048 + 64 + 16 + g.RT * g.GX * 16;
    TT_FATAL(
        port_bytes <= cap_bytes,
        "fused_tail (engine): a port core's L1 {} KB > {} KB",
        port_bytes >> 10,
        cap_bytes >> 10);
    return plan;
}

std::map<std::pair<ChipCoord, TailDir>, std::vector<uint32_t>> row_links(const MeshPlan& plan) {
    std::map<std::pair<ChipCoord, TailDir>, std::vector<uint32_t>> out;
    for (const auto& [chip, ports] : plan.chips) {
        for (TailDir d : {TailDir::Fwd, TailDir::Bwd}) {
            if (const auto& port = ports[static_cast<size_t>(mesh_dir(d))]) {
                out[{chip, d}] = port->links;
            }
        }
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
