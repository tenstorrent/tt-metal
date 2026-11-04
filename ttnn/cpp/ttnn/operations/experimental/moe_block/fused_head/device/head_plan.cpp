// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace ttnn::operations::experimental::moe_block::fused_head {

// The cost model's arithmetic with every operation rounded on its own, no fused multiply-add: the ranking of close
// layouts does not depend on what the compiler contracts.
#if defined(__clang__)
#define HEAD_PLAN_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define HEAD_PLAN_NO_FP_CONTRACT
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace head_plan_detail {

constexpr uint32_t kPortGroup = 4;     // x_gather3: a port sender's packet group
constexpr uint32_t kHeaders = 12;      // fabric packet headers a RISC
constexpr uint32_t kRouterArgs = 341;  // router_stream's per-core runtime-arg bound
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEngOverhead = 3.0;  // the logits on the matmul cores: their matmul's cost over the router cores'

using head_math::al64;
using head_math::ceil_div;
std::string kb0(double bytes) { return fmt_fixed(bytes / 1024, 0); }
std::string s(int64_t v) { return std::to_string(v); }

int64_t round_us(double v) { return static_cast<int64_t>(std::nearbyint(v)); }

enum class Cells : uint8_t { None, Column, Row, Rows };

// head_plan.router_cells: where the router cores go.
Cells router_cells(const HeadShape& shape, uint32_t GY, uint32_t GX) {
    const uint32_t nrt = shape.rows / 32 / shape.G, gy = shape.grid[1];
    if (nrt + 2 * shape.links + 1 <= gy) {
        return Cells::Column;
    }
    if (GY < gy && nrt <= GX) {
        return Cells::Row;
    }
    if (GY < gy && nrt <= GX * (gy - GY)) {
        return Cells::Rows;
    }
    return Cells::None;
}

uint32_t la_dests(uint32_t G, bool ring) { return ring && G > 2 ? G / 2 : G - 1; }

int64_t region_of(const Regions& r, const std::string& k) {
    for (const auto& [n, v] : r) {
        if (n == k) {
            return v;
        }
    }
    return 0;
}

// head_plan.cores_check: every per-core requirement past the matmul cores' arena; "" or why not.
std::string cores_check(const HeadShape& shape, const HeadLayout& lay, const RsPlan& rs, int64_t arena) {
    const uint32_t G = shape.G, L = shape.links, gy = shape.grid[1];
    const bool ring = shape.ring && G > 2;
    if (rs.P >= 256 || (G > 4 && rs.scatter != -2)) {
        return "reduce-scatter counts word: blocks of < 256 tiles (P " + s(rs.P) + "), past TP 4 the rows scatter only";
    }
    if (G > 4 && !lay.on_engine) {
        return "TP " + s(G) +
               " > 4: the router's logits on the router cores differ call to call - on the matmul cores only";
    }
    if (kPortGroup + 1 > kHeaders) {
        return "packet headers";
    }
    if (shape.latent) {
        const uint32_t nd = la_dests(G, ring);
        if (nd > 3 || kPortGroup + 1 + nd > kHeaders) {
            return "latent all-to-all: " + s(nd) + " destination blocks a port > 3 (packet headers: " + s(kHeaders) +
                   " a RISC)";
        }
    }
    const uint32_t nrt = shape.rows / 32 / G;
    if (lay.GX * lay.GY / L > 128) {
        return "a port's reduce-scatter cores: " + s(lay.GX * lay.GY / L) + " > 128 (x_gather3's rdc / seen tables)";
    }
    if (3 * L > gy) {
        return "mover column: " + s(2 * L) + " ports + " + s(L) + " copy cores > " + s(gy) + " cells";
    }
    if (nrt + 2 * L + 1 > gy) {
        const int64_t free = (int64_t(gy) - lay.GY) * lay.GX;
        const bool rect = lay.xdram && !lay.on_engine && router_rect(nrt, lay.GX, gy - lay.GY) != 0;
        if (lay.GY >= gy || !(nrt <= lay.GX || (lay.on_engine && nrt <= free) || rect)) {
            return "router cores: " + s(nrt) +
                   " (a row tile each of a chip's owned rows) fit neither the mover column (" + s(gy) + " cells less " +
                   s(2 * L) + " ports and the watcher) nor the " + s(free) + " free cells under the matmul cores (" +
                   s(lay.GX) + "x" + s(lay.GY) + (!lay.on_engine ? ", one row: the logits on the router cores" : "") +
                   ")";
        }
    }
    const uint32_t nc = nrt;
    const uint32_t ET = shape.E / 32, KT = shape.H / G / 32;
    auto layout = [&](uint32_t kbr) { return router_layout(G, ET, shape.K, kbr, ET); };
    uint32_t best = 0;
    for (uint32_t d = 1; d < 5; ++d) {
        if (KT % d == 0 && layout(d).second <= arena) {
            best = d;
        }
    }
    const auto [roff, rbytes] = layout(best ? best : 1);
    if (rbytes > arena) {
        return "router cores: " + kb0(rbytes) + " KB > the arena's " + kb0(arena) + " KB";
    }
    int64_t ns = ceil_div(nc, L);
    while (!lay.on_engine && ns > 1 && int64_t(G - 1) * ns * ET * 4096 > arena / 2) {
        --ns;
    }
    const int64_t pbuf = int64_t(G - 1) * ns * ET * 4096;
    if (pbuf > arena) {
        return "the ports' logits sub-blocks: " + kb0(pbuf) + " KB > the arena's " + kb0(arena) + " KB";
    }
    const uint32_t N = shape.N(), S = shape.S(), K = shape.K, EPC = shape.EPC(), A = shape.A;
    if (N * EPC > 1024 || S > 2048 || EPC > 64 || S % 32) {
        return "routing planner: N EPC " + s(N * EPC) + " <= 1024, S " + s(S) + " <= 2048 tokens a chip, EPC " +
               s(EPC) + " <= 64";
    }
    const int64_t rows_cap = routing_worst_rows(N, S, K, EPC);
    RoutingSizes sz{};
    int64_t step = 0, per = 0, pl0 = 0;
    uint32_t M = std::min({2 * nc, EPC, A, 14u});
    bool found = false;
    for (; M > 0; --M) {
        sz = routing_a2a_sizes(N, S, K, EPC, A, rows_cap, M);
        step = al64(sz.plan_l1);
        per = ceil_div(M, nc);
        pl0 = per * step <= region_of(roff, "bias") ? 0 : al64(rbytes);
        if (sz.fits && pl0 + per * step <= arena) {
            found = true;
            break;
        }
    }
    if (!found) {
        const std::string what =
            !sz.why.empty() ? sz.why : s(per) + " region(s) of " + kb0(step) + " KB past " + kb0(pl0) + " KB";
        return "routing planner: " + what + " > the arena's " + kb0(arena) + " KB";
    }
    const auto rt = routing_paths(shape.R, G, ring, shape.sp_ring && shape.R > 2, L);
    if (std::holds_alternative<std::string>(rt)) {
        return "routing exchange: " + std::get<std::string>(rt);
    }
    const auto [nx, ny, nf, nsend] = std::get<RoutingPaths>(rt);
    if (nf > 64) {
        return "routing exchange: a port forwards " + s(nf) + " packets > 64";
    }
    const int64_t pcap = routing_slot_bytes(S, K, EPC, M);
    if (al64(pbuf) + int64_t(nx + nf) * pcap > arena) {
        return "routing exchange: a port's buffers " + kb0(al64(pbuf) + int64_t(nx + nf) * pcap) + " KB (logits " +
               kb0(pbuf) + " + " + s(nx) + " + " + s(nf) + " packets of " + fmt_fixed(pcap / 1024.0, 1) +
               " KB) > the arena's " + kb0(arena) + " KB";
    }
    if (2 * int64_t(ny) * pcap > arena) {
        return "routing exchange: the watcher's Y buffers " + kb0(2 * int64_t(ny) * pcap) + " KB > the arena's " +
               kb0(arena) + " KB";
    }
    const int64_t plen = 26 + 5 * int64_t(M) + 3 * int64_t(N) + 3 * int64_t(nsend);
    const int64_t hand = lay.on_engine ? 0 : G - 1;
    const std::vector<std::pair<std::string, int64_t>> nargs = {
        {"BRISC", 1 + 6 * hand + 10 + 3 + 2 * int64_t(nc) + plen},
        {"NCRISC", 2 * int64_t(G) + 10 + (nc < M ? 2 + plen : 2)},
        {"compute", hand + G + 9},
    };
    std::string big;
    for (const auto& [k, v] : nargs) {
        if (v > kRouterArgs) {
            big += (big.empty() ? "" : ", ") + ("'" + k + "': ") + s(v);
        }
    }
    if (!big.empty()) {
        return "router cores' runtime args: {" + big + "} > " + s(kRouterArgs) + " (" + s(nc) + " router cores, " +
               s(M) + " planner instances, N " + s(N) + ")";
    }
    return "";
}

bool is_l1(const HeadPlanned& p) {
    return p.note.rfind("arena", 0) == 0 || p.note.rfind("router cores' arena", 0) == 0 ||
           p.note.find("of L1") != std::string::npos;
}

// head_plan._summary: why no candidate fits.
std::string summary(const std::vector<HeadPlanned>& unfit, const ArenaMax& arena_max) {
    std::vector<const HeadPlanned*> l1, other;
    for (const auto& p : unfit) {
        (is_l1(p) ? l1 : other).push_back(&p);
    }
    std::vector<std::string> out;
    if (!l1.empty()) {
        const HeadPlanned* least = nullptr;
        int64_t bestk = 0;
        for (const auto* p : l1) {
            int64_t k = (p->arena ? p->arena : kUnlimited) - arena_max(p->lay.KB);
            if (!least || k < bestk) {
                least = p;
                bestk = k;
            }
        }
        std::set<uint32_t> kbs, nchs;
        for (const auto* p : l1) {
            kbs.insert(p->lay.KB);
            nchs.insert(p->lay.NCH);
        }
        out.push_back(
            "L1: the least arena any layout needs is " + kb0(least->arena) + " KB (" + describe_layout(least->lay) +
            "), a core has " + kb0(arena_max(least->lay.KB)) + " KB; tried K steps " +
            list_str({kbs.begin(), kbs.end()}) + ", row chunks " + list_str({nchs.begin(), nchs.end()}) +
            ", W rings off / 5 / 4 / 3");
    }
    if (!other.empty()) {
        auto sorted = other;
        std::stable_sort(sorted.begin(), sorted.end(), [](const HeadPlanned* a, const HeadPlanned* b) {
            return std::make_pair(a->lay.NCH, -int64_t(a->lay.KB)) < std::make_pair(b->lay.NCH, -int64_t(b->lay.KB));
        });
        std::vector<std::string> reasons;
        const HeadPlanned* first = nullptr;
        for (const auto* p : sorted) {
            std::string r = p->note.substr(0, p->note.find(':'));
            if (std::find(reasons.begin(), reasons.end(), r) == reasons.end()) {
                reasons.push_back(r);
                if (!first) {
                    first = p;
                }
            }
        }
        std::string more = reasons.size() > 1 ? " (and " + s(reasons.size() - 1) + " other reason(s))" : std::string();
        out.push_back(first->note + " at " + describe_layout(first->lay) + more);
    }
    std::string res;
    for (size_t i = 0; i < out.size(); ++i) {
        res += (i ? "; " : "") + out[i];
    }
    return res;
}

// head_plan.degradation: what L1 forced.
std::string degradation(const HeadPlanned& chosen, const std::optional<HeadPlanned>& free) {
    HEAD_PLAN_NO_FP_CONTRACT
    if (!free || chosen.t <= free->t * 1.005) {
        return "";
    }
    const auto &a = chosen.lay, &b = free->lay;
    std::vector<std::string> d;
    if (a.KB != b.KB) {
        d.push_back("K step " + s(b.KB) + " -> " + s(a.KB));
    }
    if (a.NCH != b.NCH) {
        d.push_back("row chunks " + s(b.NCH) + " -> " + s(a.NCH));
    }
    if (a.GY != b.GY || a.MT != b.MT) {
        d.push_back("matmul core rows x MT " + s(b.GY) + "x" + s(b.MT) + " -> " + s(a.GY) + "x" + s(a.MT));
    }
    if (a.on_engine != b.on_engine) {
        auto mr = [](bool on) { return std::string(on ? "matmul" : "router"); };
        d.push_back("router logits " + mr(b.on_engine) + " cores -> " + mr(a.on_engine) + " cores");
    }
    if (a.compact != b.compact) {
        d.push_back("compact RS zones");
    }
    for (const auto& [k, v] : chosen.parts) {
        if (k == "wring" && v) {
            d.push_back("W streamed from DRAM every chunk (a ring of " + s(v) + " slots)");
        }
    }
    std::string ds;
    for (size_t i = 0; i < d.size(); ++i) {
        ds += (i ? ", " : "") + d[i];
    }
    return "L1-limited: " + (ds.empty() ? std::string("another layout") : ds) + " - predicted " +
           fmt_fixed(chosen.t, 0) + " us vs " + fmt_fixed(free->t, 0) + " us with unlimited L1 (+" +
           fmt_fixed(chosen.t - free->t, 0) + " us, +" + fmt_fixed(100 * (chosen.t / free->t - 1), 0) + "%)";
}

}  // namespace head_plan_detail

using namespace head_plan_detail;

ArenaMax arena_max_of(int64_t l1_free, uint32_t payload, int64_t cb_floor) {
    return [l1_free, payload, cb_floor](uint32_t kb) { return arena_cap_of(l1_free, kb, payload, cb_floor); };
}

ArenaMax arena_max_fixed(int64_t bytes) {
    return [bytes](uint32_t) { return bytes; };
}

uint32_t router_rect(uint32_t nrt, uint32_t cols, uint32_t rows) {
    for (int64_t c = std::min(cols, nrt); c > 1; --c) {
        if (nrt % c == 0 && nrt / c <= rows) {
            return c;
        }
    }
    return 0;
}

uint32_t engine_width(const HeadShape& shape) {
    int64_t GX = std::min<int64_t>(int64_t(shape.grid[0]) - 1, shape.I / 32);
    if (shape.latent) {
        GX = std::min<int64_t>(GX, *shape.latent / 32 / shape.G);
    }
    return GX < 0 ? 0 : uint32_t(GX);
}

std::vector<std::array<uint32_t, 3>> head_layouts(const HeadShape& shape, uint32_t mt_max) {
    const uint32_t G = shape.G, gy = shape.grid[1];
    const uint32_t GX = engine_width(shape);
    const uint32_t RT = shape.rows / 32;
    std::vector<std::array<uint32_t, 3>> out;
    if (shape.rows % 32 || RT % G) {
        return out;
    }
    const uint32_t nrt = RT / G;
    for (uint32_t nch = 1; nch <= nrt; ++nch) {
        if (nrt % nch) {
            continue;
        }
        const uint32_t CR = RT / nch;
        for (uint32_t GY = gy; GY > 0; --GY) {
            if (CR % GY || CR / GY > mt_max) {
                continue;
            }
            if (router_cells(shape, GY, GX) == Cells::None) {
                continue;
            }
            out.push_back({GY, CR / GY, nch});
        }
    }
    return out;
}

std::vector<uint32_t> k_steps(const HeadShape& shape) {
    const uint32_t klt = shape.H / 32 / shape.G;
    std::vector<uint32_t> out;
    for (uint32_t d = 16; d >= 1; --d) {
        if (klt % d == 0 && (klt / d) % 2 == 0) {
            out.push_back(d);
        }
    }
    return out;
}

std::string describe_layout(const HeadLayout& lay) {
    return "matmul cores " + s(lay.GX) + "x" + s(lay.GY) + " MT " + s(lay.MT) + ", " + s(lay.NCH) + " row chunk" +
           (lay.NCH > 1 ? "s" : "") + ", K step " + s(lay.KB) + ", router " +
           (lay.on_engine ? "on the matmul cores" : "cores");
}

HeadSized head_size(const HeadShape& shape, const HeadLayout& lay, int64_t cap) {
    std::optional<RsPlan> rs;
    std::optional<SeEngineSizing> eng;
    try {
        rs.emplace(
            shape.G,
            shape.ring && shape.G > 2,
            lay.GX,
            lay.GY,
            lay.MT,
            shape.H / 32,
            shape.rs_dim(),
            shape.links,
            kDestTiles / lay.MT,
            lay.compact);
        SeEngineArgs a;
        a.rows = shape.rows;
        a.H = shape.H;
        a.I = shape.I;
        a.GX = lay.GX;
        a.GY = lay.GY;
        a.MT = lay.MT;
        a.KB = lay.KB;
        a.shards = shape.G;
        a.rs = &*rs;
        if (shape.latent) {
            a.latent = *shape.latent / 32 / shape.G;
        }
        if (lay.on_engine) {
            a.router = std::make_pair(shape.E / 32, shape.H / shape.G / 32);
        }
        a.arena_max = cap;
        a.w_dtype = shape.w_dtype;
        a.xring = lay.xdram ? 2 : 0;
        eng.emplace(a);
    } catch (const ArenaTooLarge& err) {
        return {
            std::nullopt, err.need, {}, "arena " + kb0(err.need) + " KB > " + kb0(cap) + " KB (W ring at the least)"};
    } catch (const SizingError& err) {
        const std::string why = err.what();
        return {std::nullopt, 0, {}, why.empty() ? "sizing failed" : why};
    }
    const int64_t need = al64(router_min_bytes(shape.G, shape.E, shape.K));
    if (need > cap) {
        return {std::nullopt, need, eng->sizes, "router cores' arena " + kb0(need) + " KB > " + kb0(cap) + " KB"};
    }
    const int64_t arena = std::max(eng->arena_bytes, need);
    std::string why = cores_check(shape, lay, *rs, arena);
    if (!why.empty()) {
        return {std::nullopt, arena, eng->sizes, why};
    }
    Regions sizes = eng->sizes;
    return {std::move(eng), arena, std::move(sizes), ""};
}

std::pair<double, Parts> head_cost(
    const HeadShape& shape, const HeadLayout& lay, const SeEngineSizing& eng, const HeadCost& c) {
    HEAD_PLAN_NO_FP_CONTRACT
    const int64_t G = shape.G, MT = lay.MT, n = lay.NCH;
    const int64_t KT = shape.H / 32, HT = shape.I / 32;
    int64_t gu = 0;
    for (uint32_t p : eng.po) {
        gu = std::max<int64_t>(gu, 2 * p);
    }
    gu += shape.latent ? eng.LWMAX : 0;
    int64_t dn = 0;
    for (const auto& w : eng.wd) {
        int64_t sum = 0;
        for (uint32_t x : w) {
            sum += x;
        }
        dn = std::max(dn, sum);
    }
    const double t_gu = double(n * MT * KT * gu) * c.tile / c.clk + double(n * eng.NPG * eng.NSG) * c.round_us;
    const double t_dn = double(n * MT * HT * dn) * c.tile / c.clk + double(n * eng.NPD * eng.NSD) * c.round_us;
    const bool fhops = shape.ring && G > 2;
    const double rate = double(shape.links) * c.link_gbs * 1e3;
    double t_x, t_lat = 0.0;
    if (fhops) {
        const double hops = double(G - 1) / 2;
        t_x = std::floor(hops * shape.rows * shape.H / double(G)) * 2 / rate;
        if (shape.latent) {
            t_lat = std::floor(std::floor(hops * shape.rows * *shape.latent / double(G)) / double(G)) * 2 / rate;
        }
    } else {
        const int64_t hops = G - 1;
        t_x = double(hops * shape.rows * shape.H / G * 2) / rate;
        if (shape.latent) {
            t_lat = double(hops * shape.rows * int64_t(*shape.latent) / G / G * 2) / rate;
        }
    }
    const double t_rs = t_x;
    const int64_t w = eng.dram_w_bytes();
    const double t_w = double(w * (eng.WRING ? n : 1)) / (c.dram_gbs * 1e3);
    const double t_wn = double(n * w) / double(lay.GX) / (c.noc_gbs * 1e3);
    const double t_math = t_gu + t_dn;
    const double t_xm = double(n * eng.NPG) * (double(KT) / 2) * double(MT) * 2048 / double(std::min(eng.S0, eng.S1)) /
                        (c.xmc_gbs * 1e3);
    const double comp = std::max({t_math, t_wn, t_xm}) / double(n) + c.chunk_us * (n > 1 ? 1 : 0);
    const double steady = std::max({comp, (t_x + t_lat + t_rs) / double(n), t_w / double(n)});
    double t_se = (t_x + t_lat) / double(n) + double(n - 1) * steady + comp + t_rs / double(n);
    if (n == 1) {
        t_se = std::max(t_se, t_w + t_dn + t_rs);
    }
    double t_xd = 0.0;
    if (lay.xdram) {
        t_xd = t_x + double(int64_t(shape.rows) * shape.H * 2) / (c.dram_gbs * 1e3);
        t_se += t_xd;
    }
    const int64_t ET = shape.E / 32;
    const double t_rc = double(ET * KT) * c.rtr_tile / c.clk;
    const double chain = c.chain_us + c.chain_ek * double(ET) * double(shape.K);
    const int64_t etc = ceil_div(ET, lay.GX);
    const double t_eng = kEngOverhead * double(n) * double(MT) * double(etc) * double(KT / G) * c.rtr_tile / c.clk;
    double t_chain;
    if (lay.on_engine) {
        t_se += t_eng;
        t_chain = t_eng + chain;
    } else {
        t_chain = t_rc + chain;
    }
    const double t = c.base_us + std::max(t_se, t_chain);
    Parts parts = {
        {"gu", round_us(t_gu)},
        {"down", round_us(t_dn)},
        {"x", round_us(t_x)},
        {"rs", round_us(t_rs)},
        {"w", round_us(t_w)},
        {"wnoc", round_us(t_wn)},
        {"xmc", round_us(t_xm)},
        {"owners", int64_t(std::min(eng.S0, eng.S1))},
        {"se", round_us(t_se)},
        {"chain", round_us(t_chain)},
        {"wring", int64_t(eng.WRING)},
        {"xdram", round_us(t_xd)},
    };
    return {t, parts};
}

std::vector<HeadPlanned> head_plan_layouts(
    const HeadShape& shape, const ArenaMax& arena_max, bool keep_unfit, bool xdram) {
    std::vector<HeadPlanned> out;
    const uint32_t GXe = engine_width(shape);
    for (const auto& [GY, MT, NCH] : head_layouts(shape)) {
        for (uint32_t kb : k_steps(shape)) {
            // (both router placements - the logits on the matmul cores or on router cores - scored by the cost model)
            const Cells cells = router_cells(shape, GY, GXe);
            for (bool on : {false, true}) {
                if (cells == Cells::Rows && !on && !xdram) {
                    continue;
                }
                const std::vector<bool> compacts =
                    shape.rs_dim() == -2 && shape.G <= 4 ? std::vector<bool>{false, true} : std::vector<bool>{false};
                HeadLayout lay;
                HeadSized sized;
                for (bool compact : compacts) {
                    lay = HeadLayout{
                        .GY = GY,
                        .MT = MT,
                        .NCH = NCH,
                        .KB = kb,
                        .on_engine = on,
                        .GX = GXe,
                        .compact = compact,
                        .xdram = xdram};
                    sized = head_size(shape, lay, arena_max(kb));
                    if (sized.eng) {
                        break;
                    }
                }
                if (!sized.eng) {
                    if (keep_unfit) {
                        out.push_back(HeadPlanned{lay, sized.arena, sized.sizes, kInf, {}, sized.why});
                    }
                    continue;
                }
                auto [t, parts] = head_cost(shape, lay, *sized.eng);
                out.push_back(HeadPlanned{lay, sized.arena, sized.sizes, t, parts, ""});
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const HeadPlanned& a, const HeadPlanned& b) {
        return std::make_pair(a.t, -int64_t(a.lay.KB)) < std::make_pair(b.t, -int64_t(b.lay.KB));
    });
    return out;
}

std::string shape_checks(const HeadShape& shape) {
    const uint32_t G = shape.G, H = shape.H;
    if (G < 2) {
        return "TP " + s(G) + ": the fused head gathers x over the TP axis (TP >= 2)";
    }
    if (G > 4 && shape.latent) {
        return "TP " + s(G) +
               " > 4 with a latent: the latent all-to-all sends a port's destination blocks under one packet header "
               "each - at most 3 (x_gather3 la_block) - a mesh with TP 2 or 4";
    }
    if (G > 8) {
        return "TP " + s(G) + " > 8: the routing / combine port tables (a mesh with TP <= 8)";
    }
    if (shape.links != 1 && shape.links != 2) {
        return s(shape.links) + " links: 1 or 2 (a port per direction and link)";
    }
    if (H % (32 * G)) {
        return "H " + s(H) + " is not whole tiles over TP " + s(G) + " (a multiple of " + s(32 * G) + ")";
    }
    const uint32_t klt = H / 32 / G;
    if (k_steps(shape).empty()) {
        return "H / 32 / TP = " + s(klt) +
               " tiles a shard: no K step <= 16 gives an even count a shard (an even shard)";
    }
    if (shape.I < 32 || shape.I % 32) {
        return "the shared expert's I / TP = " + s(shape.I) + ": whole tiles, at least one (I_s a multiple of " +
               s(32 * G) + ")";
    }
    if (shape.latent && (*shape.latent % (32 * G))) {
        return "latent " + s(*shape.latent) + " is not whole tiles over TP " + s(G) + " (a multiple of " + s(32 * G) +
               ")";
    }
    if (shape.K % 2 || shape.K > 16) {
        return "top-K " + s(shape.K) +
               ": even and <= 16 (the joint routing rows - K ids + K weights in 64 B - dispatch_prep, routing_a2a)";
    }
    if (shape.E % 32) {
        return s(shape.E) + " router candidates: whole tiles (a multiple of 32)";
    }
    if (shape.rows % (32 * G)) {
        const int64_t nxt = ceil_div(shape.rows, 32 * G) * 32 * G;
        return "rows a mesh row " + s(shape.rows) + " must be a multiple of 32 x TP = " + s(32 * G) +
               "; nearest valid " + s(nxt);
    }
    const uint32_t N = shape.N(), S = shape.S(), EPC = shape.EPC();
    if (N > 32 || shape.A > 16) {
        return "routing planner: N " + s(N) + " <= 32 chips, accumulators " + s(shape.A) + " <= 16";
    }
    if (S > 2048) {
        return "routing planner: " + s(S) + " tokens a chip > 2048 (an entry's token field); chunk <= " + s(2048 * N) +
               " tokens";
    }
    if (EPC > 64 || N * EPC > 1024) {
        return "routing planner: " + s(EPC) + " experts a chip <= 64 and N x EPC " + s(N * EPC) +
               " <= 1024 (fold the router)";
    }
    if (engine_width(shape) < 1) {
        return "no matmul core column: I / TP " + s(shape.I) + " or latent tiles a chip";
    }
    return "";
}

HeadFit head_plan_fit(const HeadShape& shape, const HeadFitOptions& options) {
    return head_plan_fit(
        shape, arena_max_of(options.l1_free.value_or(kL1Free - kL1Margin), shape.payload, options.cb_floor), options);
}

HeadFit head_plan_fit(const HeadShape& shape, const ArenaMax& arena_max, const HeadFitOptions& options) {
    HeadFit res;
    res.why = shape_checks(shape);
    if (!res.why.empty()) {
        return res;
    }
    if (head_layouts(shape).empty()) {
        const uint32_t nrt = shape.rows / 32 / shape.G;
        res.why = "no matmul-core layout for " + s(shape.rows) + " rows a mesh row (" + s(nrt) +
                  " row tiles an owner: row chunks of whole owner blocks, MT <= 2, the router cores' cells)";
        return res;
    }
    const auto ps = head_plan_layouts(shape, arena_max, true);
    std::vector<HeadPlanned> ok;
    for (const auto& p : ps) {
        if (p.t != kInf) {
            ok.push_back(p);
        }
    }
    if (ok.empty()) {
        std::vector<HeadPlanned> unfit;
        for (const auto& p : ps) {
            if (p.t == kInf) {
                unfit.push_back(p);
            }
        }
        if (unfit.empty()) {
            res.why = "no layout";
            return res;
        }
        std::string why = summary(unfit, arena_max);
        if (options.fallback) {
            const auto pxa = head_plan_layouts(shape, arena_max, true, true);
            std::vector<HeadPlanned> px, ux;
            for (const auto& p : pxa) {
                if (p.t != kInf) {
                    px.push_back(p);
                }
                if (p.t == kInf) {
                    ux.push_back(p);
                }
            }
            if (px.empty() && !ux.empty()) {
                why += "; the DRAM fallback (x own a ring of 2 chunks) does not fit either - " + summary(ux, arena_max);
            }
            if (!px.empty()) {
                int64_t xd = 0;
                for (const auto& [k, v] : px[0].parts) {
                    if (k == "xdram") {
                        xd = v;
                    }
                }
                res.ok = true;
                res.note = "x from DRAM (the fallback: " + why.substr(0, why.find(';')) +
                           "): x gathered over TP ahead of the program, x own a ring of 2 chunks reloaded from it - "
                           "predicted " +
                           fmt_fixed(px[0].t, 0) + " us, the gather and the reloads " + s(xd) + " us of it";
                res.planned = std::move(px);
                return res;
            }
        }
        if (options.largest && why.find("L1:") != std::string::npos) {
            const uint32_t step = 32 * shape.G;
            int64_t lo = 0, hi = int64_t(shape.rows / step) - 1;
            HeadFitOptions sub = options;
            sub.largest = false;
            sub.explain = false;
            while (lo < hi) {
                const int64_t mid = (lo + hi + 1) / 2;
                HeadShape smaller = shape;
                smaller.rows = uint32_t(mid * step);
                if (head_plan_fit(smaller, arena_max, sub).ok) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            if (lo) {
                why += "; the largest rows a mesh row that fit: " + s(lo * step) + " (chunk " + s(lo * step * shape.R) +
                       ")";
            }
        }
        res.why = why;
        return res;
    }
    res.ok = true;
    res.planned = ok;
    if (!options.explain) {
        return res;
    }
    try {
        const auto free = head_plan_layouts(shape, arena_max_fixed(kUnlimited), false);
        if (!free.empty()) {
            res.free = free.front();
        }
    } catch (const std::exception&) {  // (the unconstrained pick only explains the note)
        res.free.reset();
    }
    res.note = degradation(res.planned[0], res.free);
    return res;
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head

#if !defined(__clang__)
#pragma GCC pop_options
#endif
#undef HEAD_PLAN_NO_FP_CONTRACT
