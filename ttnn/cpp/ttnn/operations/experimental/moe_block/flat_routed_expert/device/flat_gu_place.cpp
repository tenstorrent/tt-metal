// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_gu_place.hpp"

#include "flat_geometry.hpp"
#include "flat_routed_expert_plan.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <tuple>

#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace gu_place_detail {
using Core = tt::tt_metal::CoreCoord;
constexpr int NX = 17, NY = 12;            // the Blackhole NoC torus
constexpr uint32_t NLINK = 2 * NX * NY * 2;  // (noc, x, y, first / second direction)

uint32_t link(uint32_t noc, int x, int y, uint32_t d) { return ((noc * NX + x) * NY + y) * 2 + d; }

// NOC0: east (x + 1) then south (y + 1); NOC1: north (y - 1) then west (x - 1)
void route(const Core& s, const Core& d, uint32_t noc, std::vector<uint32_t>& out) {
    int x = int(s.x), y = int(s.y);
    const int dx = int(d.x), dy = int(d.y);
    if (noc == 0) {
        while (x != dx) {
            out.push_back(link(0, x, y, 0));
            x = (x + 1) % NX;
        }
        while (y != dy) {
            out.push_back(link(0, x, y, 1));
            y = (y + 1) % NY;
        }
    } else {
        while (y != dy) {
            out.push_back(link(1, x, y, 0));
            y = (y + NY - 1) % NY;
        }
        while (x != dx) {
            out.push_back(link(1, x, y, 1));
            x = (x + NX - 1) % NX;
        }
    }
}

uint32_t hops(const Core& s, const Core& d, uint32_t noc) {
    std::vector<uint32_t> r;
    route(s, d, noc, r);
    return r.size();
}

// how -> the NoC a core's reads take (their responses leave the bank from its endpoint tile of that NoC: banks[b][noc])
uint32_t noc_of(uint32_t how) { return how == FlatGuPlace::tensix_noc1 ? 1u : 0u; }

struct Model {
    const FlatGuGeom* g;
    std::vector<double> bg;
    // routes[(cell * banks + bank) * 3 + how]
    std::vector<std::vector<uint32_t>> routes;
    uint32_t nb = 0;

    Model(const FlatGuGeom& geom, const FlatGuTraffic& t) : g(&geom), bg(NLINK, 0.0), nb(geom.banks.size()) {
        std::vector<uint32_t> r;
        for (const auto& h : geom.helpers) {
            for (const auto& b : geom.banks) {
                r.clear();
                route(b[1], h, 1, r);
                for (uint32_t l : r) {
                    bg[l] += t.x / geom.helpers.size() / nb;
                }
            }
        }
        for (const auto& dc : geom.down) {
            for (const auto& b : geom.banks) {
                r.clear();
                route(b[0], dc, 0, r);
                for (uint32_t l : r) {
                    bg[l] += t.down / geom.down.size() / nb;
                }
            }
        }
        routes.resize(geom.cells.size() * nb * 3);
        for (uint32_t c = 0; c < geom.cells.size(); ++c) {
            for (uint32_t b = 0; b < nb; ++b) {
                for (const uint32_t how : {FlatGuPlace::tensix_noc0, FlatGuPlace::tensix_noc1}) {
                    const uint32_t noc = noc_of(how);
                    route(geom.banks[b][noc], geom.cells[c], noc, routes[(c * nb + b) * 3 + how]);
                }
            }
        }
    }
    const std::vector<uint32_t>& of(uint32_t c, uint32_t b, uint32_t how) const { return routes[(c * nb + b) * 3 + how]; }
};

struct State {
    std::vector<double> load;
    double sumsq = 0.0;
    void add(const std::vector<uint32_t>& r, double v) {
        for (uint32_t l : r) {
            const double o = load[l];
            load[l] = o + v;
            sumsq += load[l] * load[l] - o * o;
        }
    }
    double max() const { return *std::max_element(load.begin(), load.end()); }
    // the summed squares of the TOPK busiest links (the tie-break: flattens the heaviest links rather than shortening
    // every route)
    double top_sq() const {
        constexpr size_t TOPK = 40;
        std::vector<double> v(load);
        std::nth_element(v.begin(), v.begin() + TOPK, v.end(), std::greater<double>());
        double q = 0.0;
        for (size_t i = 0; i < TOPK; ++i) {
            q += v[i] * v[i];
        }
        return q;
    }
};

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}
}  // namespace gu_place_detail
using namespace gu_place_detail;

std::vector<uint32_t> FlatGuGeom::key() const {
    std::vector<uint32_t> k{static_cast<uint32_t>(cells.size()), static_cast<uint32_t>(banks.size())};
    auto put = [&](const Core& c) { k.push_back(static_cast<uint32_t>((c.x << 16) | c.y)); };
    for (const auto& c : cells) {
        put(c);
    }
    for (const auto& b : banks) {
        for (const auto& c : b) {
            put(c);
        }
    }
    for (const auto* l : {&helpers, &down}) {
        k.push_back(static_cast<uint32_t>(l->size()));
        for (const auto& c : *l) {
            put(c);
        }
    }
    return k;
}

std::vector<Core> flat_gu_cells(const FlatRoutedExpertPlan& p) {
    std::vector<Core> out;
    for (const auto& [x0, x1, y0, y1] : p.rects) {
        for (uint32_t y = y0; y <= y1; ++y) {
            for (uint32_t x = x0; x <= x1; ++x) {
                out.emplace_back(x, y);
            }
        }
    }
    return out;
}

FlatGuGeom flat_gu_geom(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertPlan& p,
    const tt::tt_metal::distributed::MeshCoordinate& coord) {
    FlatGuGeom g;
    for (const auto& c : flat_gu_cells(p)) {
        g.cells.push_back(flat_worker_noc0(mesh_device, c, coord));
    }
    const FlatDramGeometry dram = FlatDramGeometry::of(mesh_device, coord);
    for (uint32_t b = 0; b < dram.banks; ++b) {
        g.banks.push_back({dram.endpoint(b, 0), dram.endpoint(b, 1)});
    }
    for (size_t i = 2; i < p.relays.size(); ++i) {  // (relays: 2 primaries, then the helpers that read x)
        g.helpers.push_back(flat_worker_noc0(mesh_device, p.relays[i], coord));
    }
    for (const auto& c : p.down) {
        g.down.push_back(flat_worker_noc0(mesh_device, c, coord));
    }
    return g;
}

FlatGuLoad flat_gu_load(const FlatGuGeom& g, const FlatGuAssign& a, const FlatGuTraffic& t) {
    const Model m(g, t);
    State s{m.bg, 0.0};
    for (double v : s.load) {
        s.sumsq += v * v;
    }
    for (uint32_t c = 0; c < g.cells.size(); ++c) {
        s.add(m.of(c, a.bank[c], a.how[c]), t.gu_core);
    }
    return FlatGuLoad{s.max(), s.top_sq()};
}

namespace gu_place_detail {
FlatGuAssign search_once(
    const std::vector<const FlatGuGeom*>& geoms,
    const FlatGuTraffic& t,
    uint32_t rg,
    uint32_t iters,
    uint64_t seed,
    std::pair<double, double>& out_score);
}

FlatGuAssign flat_gu_search(
    const std::vector<const FlatGuGeom*>& geoms,
    const FlatGuTraffic& t,
    uint32_t rg,
    uint32_t iters,
    uint32_t restarts) {
    // (restarts: the same greedy start, different move sequences; the local search stops in different minima)
    FlatGuAssign best;
    std::pair<double, double> bs{1e300, 1e300};
    for (uint32_t r = 0; r < restarts; ++r) {
        std::pair<double, double> sc;
        FlatGuAssign a = search_once(geoms, t, rg, iters, 0x9E3779B97F4A7C15ull + 0x1234567ull * r, sc);
        if (sc < bs) {
            bs = sc;
            best = std::move(a);
        }
    }
    return best;
}

FlatGuAssign gu_place_detail::search_once(
    const std::vector<const FlatGuGeom*>& geoms,
    const FlatGuTraffic& t,
    uint32_t rg,
    uint32_t iters,
    uint64_t seed,
    std::pair<double, double>& out_score) {
    TT_FATAL(!geoms.empty(), "flat gate/up placement: no geometry");
    const FlatGuGeom& g0 = *geoms.front();
    const uint32_t nc = g0.cells.size(), nb = g0.banks.size();
    TT_FATAL(nc == 2 * rg * nb, "flat gate/up placement: {} cells for {} banks x 2 x {} cores", nc, nb, rg);
    for (const auto* g : geoms) {
        TT_FATAL(g->cells.size() == nc && g->banks.size() == nb, "flat gate/up placement: geometries differ in shape");
    }
    // start: greedy by hops on the first geometry (either NoC, 2 RG cores a bank)
    FlatGuAssign a{std::vector<uint32_t>(nc, 0), std::vector<uint32_t>(nc, 0)};
    std::vector<bool> taken(nc, false);
    auto fill = [&](const std::vector<uint32_t>& hows, uint32_t cap) {
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> pairs;  // (hops, bank, cell, how)
        for (uint32_t b = 0; b < nb; ++b) {
            for (uint32_t c = 0; c < nc; ++c) {
                if (taken[c]) {
                    continue;
                }
                uint32_t best = ~0u, bh = 0;
                for (uint32_t how : hows) {
                    const uint32_t noc = noc_of(how);
                    const uint32_t h = hops(g0.banks[b][noc], g0.cells[c], noc);
                    if (h < best) {
                        best = h;
                        bh = how;
                    }
                }
                pairs.emplace_back(best, b, c, bh);
            }
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<uint32_t> used(nb, 0);
        for (const auto& [h, b, c, how] : pairs) {
            if (!taken[c] && used[b] < cap) {
                taken[c] = true;
                ++used[b];
                a.bank[c] = b;
                a.how[c] = how;
            }
        }
    };
    fill({FlatGuPlace::tensix_noc0, FlatGuPlace::tensix_noc1}, 2 * rg);
    // local search: swap two cells' (bank, how), flip a cell's NoC; keep a move that does not worsen (the worst chip's
    // busiest link, the summed squares)
    std::vector<Model> models;
    std::vector<State> st;
    for (const auto* g : geoms) {
        models.emplace_back(*g, t);
        State s{models.back().bg, 0.0};
        for (double v : s.load) {
            s.sumsq += v * v;
        }
        for (uint32_t c = 0; c < nc; ++c) {
            s.add(models.back().of(c, a.bank[c], a.how[c]), t.gu_core);
        }
        st.push_back(std::move(s));
    }
    // (the worst chip's busiest link, then the summed squares of every link: a top-40 tie-break measured no better
    // on K3 and cost 3x the search time)
    auto worst = [&]() {
        double mx = 0.0;
        for (const auto& s : st) {
            mx = std::max(mx, s.max());
        }
        return mx;
    };
    auto ties = [&]() {
        double sq = 0.0;
        for (const auto& s : st) {
            sq += s.sumsq;
        }
        return sq;
    };
    auto score = [&]() { return std::make_pair(worst(), ties()); };
    auto best = score();
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> changed;  // (cell, old bank, old how)
    auto set = [&](uint32_t c, uint32_t b, uint32_t how) {
        changed.emplace_back(c, a.bank[c], a.how[c]);
        for (size_t i = 0; i < models.size(); ++i) {
            st[i].add(models[i].of(c, a.bank[c], a.how[c]), -t.gu_core);
            st[i].add(models[i].of(c, b, how), t.gu_core);
        }
        a.bank[c] = b;
        a.how[c] = how;
    };
    for (uint32_t it = 0; it < iters; ++it) {
        changed.clear();
        const uint32_t u = xorshift(seed) % 100;
        if (u < 60) {
            const uint32_t i = xorshift(seed) % nc, j = xorshift(seed) % nc;
            if (i == j || (a.bank[i] == a.bank[j] && a.how[i] == a.how[j])) {
                continue;
            }
            const uint32_t bi = a.bank[i], hi = a.how[i], bj = a.bank[j], hj = a.how[j];
            set(i, bj, hj);
            set(j, bi, hi);
        } else {
            const uint32_t i = xorshift(seed) % nc;
            set(i, a.bank[i], a.how[i] == FlatGuPlace::tensix_noc0 ? FlatGuPlace::tensix_noc1 : FlatGuPlace::tensix_noc0);
        }
        const double mx = worst();
        if (mx < best.first) {
            best = {mx, ties()};
            continue;
        }
        if (mx == best.first) {
            const double sq = ties();
            if (sq <= best.second) {
                best.second = sq;
                continue;
            }
        }
        for (auto ci = changed.rbegin(); ci != changed.rend(); ++ci) {  // undo
            const auto [c, b, how] = *ci;
            for (size_t i = 0; i < models.size(); ++i) {
                st[i].add(models[i].of(c, a.bank[c], a.how[c]), -t.gu_core);
                st[i].add(models[i].of(c, b, how), t.gu_core);
            }
            a.bank[c] = b;
            a.how[c] = how;
        }
    }
    out_score = best;
    return a;
}

std::vector<uint32_t> flat_gu_words(const FlatRoutedExpertPlan& p, const FlatGuAssign& a) {
    const auto cells = flat_gu_cells(p);
    std::vector<uint32_t> w;
    for (uint32_t c = 0; c < cells.size(); ++c) {
        w.push_back(FlatGuPlace::word(
            cells[c].x, cells[c].y, a.bank[c], static_cast<FlatGuPlace::How>(a.how[c])));
    }
    return w;
}

FlatGuAssign flat_gu_from_words(const FlatRoutedExpertPlan& p, const std::vector<uint32_t>& words) {
    const auto cells = flat_gu_cells(p);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> at;
    for (uint32_t c = 0; c < cells.size(); ++c) {
        at[{cells[c].x, cells[c].y}] = c;
    }
    FlatGuAssign a{std::vector<uint32_t>(cells.size(), 0), std::vector<uint32_t>(cells.size(), 0)};
    for (const uint32_t w : words) {
        const auto it = at.find({w >> 24, (w >> 16) & 0xFF});
        TT_FATAL(it != at.end(), "flat gate/up placement: word {:#x} names no gate/up cell", w);
        a.bank[it->second] = (w >> 8) & 0xFF;
        a.how[it->second] = w & 0xFF;
    }
    return a;
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
