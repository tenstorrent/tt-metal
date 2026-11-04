// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_sizing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <set>
#include <tuple>

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace head_sizing_detail {

using head_math::al16;
using head_math::al64;
using head_math::ceil_div;
int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

[[noreturn]] void fail(const std::string& why) { throw SizingError(why); }

// se_engine3._split: n tiles into `parts` widths, multiples of unit, as even as possible (each >= unit).
std::vector<uint32_t> split_units(uint32_t n, uint32_t parts, uint32_t unit) {
    if (!(n % unit == 0 && n / unit >= parts)) {
        fail("(" + std::to_string(n) + ", " + std::to_string(parts) + ", " + std::to_string(unit) + ")");
    }
    uint32_t base = n / unit / parts, rem = n / unit % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = (base + (i < rem ? 1 : 0)) * unit;
    }
    return out;
}

std::vector<uint32_t> split_pad(uint32_t n, uint32_t parts, uint32_t unit) {
    return n / unit >= parts ? split_units(n, parts, unit) : std::vector<uint32_t>(parts, unit);
}

// se_rs._split: n into `parts` counts, as even as possible (zeros allowed).
std::vector<uint32_t> split_even(uint32_t n, uint32_t parts) {
    uint32_t base = n / parts, rem = n % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = base + (i < rem ? 1 : 0);
    }
    return out;
}

std::vector<uint32_t> prefix(const std::vector<uint32_t>& v) {
    std::vector<uint32_t> out(v.size());
    uint32_t s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = s;
        s += v[i];
    }
    return out;
}

uint32_t vmax(const std::vector<uint32_t>& v) { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
uint32_t vmin(const std::vector<uint32_t>& v) { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
uint32_t vsum(const std::vector<uint32_t>& v) { return std::accumulate(v.begin(), v.end(), 0u); }

// se_engine3._subsets: two owner-subset sizes dividing nf, covering as many of GX columns as possible (the first
// maximum in (a, b) order, as Python's max).
std::pair<uint32_t, uint32_t> subsets(uint32_t nf, uint32_t GX) {
    std::vector<uint32_t> divs;
    for (uint32_t d = 1; d <= GX; ++d) {
        if (nf % d == 0) {
            divs.push_back(d);
        }
    }
    auto key = [&](uint32_t a, uint32_t b) { return std::make_pair(std::min(a + b, GX), -std::abs(int(a) - int(b))); };
    std::pair<uint32_t, uint32_t> best{divs[0], divs[0]};
    for (uint32_t a : divs) {
        for (uint32_t b : divs) {
            if (key(a, b) > key(best.first, best.second)) {
                best = {a, b};
            }
        }
    }
    return best;
}

std::string nested_str(const std::vector<std::vector<uint32_t>>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? ", " : "") + list_str(v[i]);
    }
    return s + "]";
}

int64_t region(const Regions& r, const std::string& k) {
    for (const auto& [n, v] : r) {
        if (n == k) {
            return v;
        }
    }
    throw SizingError("'" + k + "'");
}

void set_region(Regions& r, const std::string& k, int64_t v) {
    for (auto& [n, x] : r) {
        if (n == k) {
            x = v;
            return;
        }
    }
    r.emplace_back(k, v);
}

}  // namespace head_sizing_detail

using namespace head_sizing_detail;

std::string fmt_fixed(double v, int precision) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return buf;
}

std::string list_str(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s + "]";
}

int64_t port_cb_bytes(uint32_t kb, uint32_t payload, uint32_t group) {
    int64_t rp = std::max<int64_t>(1, std::min<int64_t>(kb, payload / 2048));
    return 2 * int64_t(group) * rp * 2048;
}

int64_t arena_cap_of(int64_t l1_free, uint32_t kb, uint32_t payload, int64_t cb_floor) {
    return std::min(kArenaMax, floor_div(l1_free - std::max(port_cb_bytes(kb, payload), cb_floor) - 1024, 64) * 64);
}

std::vector<uint32_t> column_split(uint32_t n, uint32_t parts, uint32_t unit) {
    if (n % unit) {
        fail("(" + std::to_string(n) + ", " + std::to_string(unit) + ")");
    }
    uint32_t base = n / unit / parts, rem = n / unit % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = (base + (i < rem ? 1 : 0)) * unit;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// plan_transport

RsKind rs_kind(uint32_t G, bool ring) {
    if (!ring || G == 2) {
        return RsKind::Line;
    }
    return G % 2 == 0 ? RsKind::RingBalanced : RsKind::RingUnbalanced;
}

namespace head_sizing_detail {

int64_t pymod(int64_t a, int64_t b) { return ((a % b) + b) % b; }

RsRole rs_role(uint32_t p, uint32_t b, uint32_t h, uint32_t G, RsKind kind) {
    if (b == p) {
        return {RsDir::Fin, false, 0};
    }
    if (kind == RsKind::Line) {
        if (b > p) {
            return {RsDir::Fwd, p == 0, b - p};
        }
        return {RsDir::Bwd, p == G - 1, p - b};
    }
    uint32_t df = pymod(int64_t(b) - p, G), db = G - df;
    if (kind == RsKind::RingUnbalanced) {
        uint32_t nf = G / 2, nb = G - 1 - nf;
        if (df <= nf) {
            return {RsDir::Fwd, df == nf, df};
        }
        return {RsDir::Bwd, db == nb, db};
    }
    uint32_t H = G / 2;
    if (h == 0) {
        if (df <= H) {
            return {RsDir::Fwd, df == H, df};
        }
        return {RsDir::Bwd, db == H - 1, db};
    }
    if (df <= H - 1) {
        return {RsDir::Fwd, df == H - 1, df};
    }
    return {RsDir::Bwd, db == H, db};
}

}  // namespace head_sizing_detail

RsTransport plan_transport(uint32_t G, RsKind kind) {
    const bool ring = kind != RsKind::Line;
    RsTransport out;
    out.chips.resize(G);
    for (uint32_t p = 0; p < G; ++p) {
        auto& sc = out.chips[p];
        for (uint32_t b = 0; b < G; ++b) {
            for (uint32_t h = 0; h < 2; ++h) {
                sc.roles[{b, h}] = rs_role(p, b, h, G, kind);
            }
        }
        // build_schedule (no mirror): the other blocks by descending distance (ties fwd first, then lower b), own last
        std::vector<uint32_t> others;
        for (uint32_t b = 0; b < G; ++b) {
            if (b != p) {
                others.push_back(b);
            }
        }
        auto key = [&](uint32_t b) {
            uint32_t k = std::max(sc.roles[{b, 0}].dist, sc.roles[{b, 1}].dist);
            return std::make_tuple(-int64_t(k), sc.roles[{b, 0}].dir == RsDir::Fwd ? 0 : 1, b);
        };
        std::stable_sort(others.begin(), others.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
        others.push_back(p);
        sc.entries[0] = others;
        sc.entries[1] = others;
    }
    auto up = [&](uint32_t p, RsDir d) -> std::optional<uint32_t> {
        int64_t q = d == RsDir::Fwd ? int64_t(p) - 1 : int64_t(p) + 1;
        if (ring) {
            return uint32_t(pymod(q, G));
        }
        if (q >= 0 && q < G) {
            return uint32_t(q);
        }
        return std::nullopt;
    };
    for (uint32_t p = 0; p < G; ++p) {
        auto& sc = out.chips[p];
        for (uint32_t h = 0; h < 2; ++h) {
            for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
                auto u = up(p, d);
                std::vector<uint32_t> lst;
                for (uint32_t b : sc.entries[h]) {
                    const auto& r = sc.roles.at({b, h});
                    if (r.dir == d && !r.start) {
                        lst.push_back(b);
                    } else if (r.dir == RsDir::Fin && u && out.chips[*u].roles.at({b, h}).dir == d) {
                        lst.push_back(b);
                    }
                }
                sc.arrivals[{h, d}] = lst;
            }
        }
    }
    for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
        size_t m = 0;
        for (const auto& sc : out.chips) {
            for (uint32_t h = 0; h < 2; ++h) {
                m = std::max(m, sc.arrivals.at({h, d}).size());
            }
        }
        out.slots[static_cast<int>(d)] = std::max<uint32_t>(1, m);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// RsPlan

RsPlan::RsPlan(
    uint32_t G_,
    bool ring,
    uint32_t GX_,
    uint32_t GY_,
    uint32_t MT_,
    uint32_t NTH_,
    int scatter_,
    uint32_t nv_,
    uint32_t pwmax_,
    bool compact_) :
    G(G_),
    GX(GX_),
    GY(GY_),
    MT(MT_),
    NTH(NTH_),
    nv(nv_),
    pwmax(pwmax_),
    CR(GY_ * MT_),
    ncores(GX_ * GY_),
    scatter(scatter_),
    compact(compact_),
    kind(rs_kind(G_, ring)),
    transport(plan_transport(G_, kind)) {
    if (scatter == -1) {
        if (NTH % G) {
            fail("");
        }
        NB = NTH / G;
        w = split_even(NB, GX);
        if (vmin(w) < 1) {
            fail("");
        }
        s = prefix(w);
        for (uint32_t x : w) {
            npb = std::max<uint32_t>(npb, ceil_div(x, pwmax));
        }
        P = MT * vmax(w);
    } else {
        if (CR % G) {
            fail("");
        }
        RB = CR / G;
        pn = split_even(NTH, GX);
        n0 = prefix(pn);
        for (uint32_t x : pn) {
            npass = std::max<uint32_t>(npass, ceil_div(x, pwmax));
        }
        P = MT * vmax(pn);
    }
    pm = scatter == -2;
    if (scatter == -2 && kind == RsKind::RingBalanced && GX > 1) {
        uint32_t best = 0;
        int64_t bestv = -1;
        for (uint32_t i = 0; i + 1 < GX; ++i) {
            int64_t pre = 0;
            for (uint32_t c = 0; c <= i; ++c) {
                pre += pn[c];
            }
            int64_t v = std::abs(2 * pre - int64_t(NTH));
            if (bestv < 0 || v < bestv) {
                bestv = v;
                best = i;
            }
        }
        hcol = 1 + best;
    }
}

uint32_t RsPlan::half(uint32_t k) const {
    if (hcol) {
        return k % GX < *hcol ? 0 : 1;
    }
    return k < ceil_div(ncores, 2) ? 0 : 1;
}

std::pair<uint32_t, uint32_t> RsPlan::rows_in(uint32_t cy, uint32_t b) const {
    uint32_t first = 0, n = 0;
    for (uint32_t r = 0; r < MT; ++r) {
        if ((cy * MT + r) / RB == b) {
            if (n == 0) {
                first = r;
            }
            ++n;
        }
    }
    return {n ? first : 0, n};
}

uint32_t RsPlan::count(uint32_t cx, uint32_t cy, uint32_t b) const {
    if (scatter == -1) {
        return MT * w[cx];
    }
    return rows_in(cy, b).second * pn[cx];
}

const RsRole& RsPlan::role(uint32_t p, uint32_t b, uint32_t h) const { return transport.chips[p].roles.at({b, h}); }

std::vector<RsPlan::Pass> RsPlan::column_passes(uint32_t p, uint32_t cx) const {
    std::vector<Pass> out;
    if (scatter == -1) {
        const auto& ents = entries(p);
        for (uint32_t e = 0; e < ents.size(); ++e) {
            uint32_t c0 = ents[e] * NB + s[cx];
            for (uint32_t pw : split_even(w[cx], npb)) {
                out.push_back({c0, pw, e});
                c0 += pw;
            }
        }
    } else {
        uint32_t c0 = n0[cx];
        for (uint32_t pw : split_even(pn[cx], npass)) {
            out.push_back({c0, pw, std::nullopt});
            c0 += pw;
        }
    }
    return out;
}

uint32_t RsPlan::finb_tiles(uint32_t cx, uint32_t cy) const {
    uint32_t n = 0;
    for (uint32_t b = 0; b < G; ++b) {
        n += count(cx, cy, b);
    }
    return n;
}

bool RsPlan::needs_hold(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const {
    uint32_t h = half(k(cx, cy));
    const auto& r = role(p, entries(p, h)[e], h);
    return !(r.dir != RsDir::Fin && r.start);
}

namespace head_sizing_detail {

uint32_t index_of(const std::vector<uint32_t>& v, uint32_t x) {
    auto it = std::find(v.begin(), v.end(), x);
    if (it == v.end()) {
        fail(std::to_string(x) + " is not in list");
    }
    return it - v.begin();
}

}  // namespace head_sizing_detail

std::pair<std::map<uint32_t, uint32_t>, uint32_t> RsPlan::hold_offs(uint32_t p, uint32_t cx, uint32_t cy) const {
    const auto o = ops(p, cx, cy);
    uint32_t h = half(k(cx, cy));
    const auto& ents = entries(p, h);
    const auto passes = column_passes(p, cx);
    std::vector<std::pair<uint32_t, uint32_t>> first;  // (entry, first op), insertion order
    auto has_first = [&](uint32_t e) {
        return std::any_of(first.begin(), first.end(), [&](const auto& f) { return f.first == e; });
    };
    for (uint32_t oi = 0; oi < o.size(); ++oi) {
        if (o[oi].first != Op::Pass) {
            continue;
        }
        const auto& ps = passes[o[oi].second];
        std::vector<uint32_t> es;
        if (ps.entry) {
            es.push_back(*ps.entry);
        } else {
            for (uint32_t r = 0; r < MT; ++r) {
                es.push_back(index_of(ents, (cy * MT + r) / RB));
            }
        }
        for (uint32_t e2 : es) {
            if (!has_first(e2)) {
                first.emplace_back(e2, oi);
            }
        }
    }
    std::map<uint32_t, uint32_t> end;
    for (uint32_t oi = 0; oi < o.size(); ++oi) {
        if (o[oi].first == Op::Fin) {
            end[o[oi].second] = oi;
        }
    }
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> live;  // (offset, tiles, entry)
    std::map<uint32_t, uint32_t> offs;
    uint32_t top = 0;
    for (uint32_t oi = 0; oi < o.size(); ++oi) {
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> keep;
        for (const auto& l : live) {
            auto it = end.find(std::get<2>(l));
            if (it == end.end()) {
                fail(std::to_string(std::get<2>(l)));
            }
            if (it->second >= oi) {
                keep.push_back(l);
            }
        }
        live = keep;
        for (const auto& [e, f] : first) {
            if (f == oi && needs_hold(p, cx, cy, e) && count(cx, cy, ents[e]) > 0) {
                uint32_t n = count(cx, cy, ents[e]);
                uint32_t off = 0;
                auto sorted_live = live;
                std::sort(sorted_live.begin(), sorted_live.end());
                for (const auto& [lo, ln, _] : sorted_live) {
                    if (off + n <= lo) {
                        break;
                    }
                    off = std::max(off, lo + ln);
                }
                live.emplace_back(off, n, e);
                offs[e] = off;
                top = std::max(top, off + n);
            }
        }
    }
    return {offs, top};
}

std::optional<uint32_t> RsPlan::down(uint32_t p, RsDir d) const {
    int64_t q = d == RsDir::Fwd ? int64_t(p) + 1 : int64_t(p) - 1;
    if (kind != RsKind::Line) {
        return uint32_t(pymod(q, G));
    }
    if (q >= 0 && q < G) {
        return uint32_t(q);
    }
    return std::nullopt;
}

std::optional<uint32_t> RsPlan::piece_entry(uint32_t p, uint32_t h, RsDir d) const {
    if (!pm || !down(p, d)) {
        return std::nullopt;
    }
    // port_units(p, d)'s first unit: (entry, half mask, pieces)
    const auto& ents = entries(p, 0);
    for (uint32_t e = 0; e < ents.size(); ++e) {
        uint32_t b = ents[e], mask = 0;
        for (uint32_t hh = 0; hh < 2; ++hh) {
            if (role(p, b, hh).dir == d) {
                mask |= 1u << hh;
            }
        }
        if (!mask) {
            continue;
        }
        bool pieces = true;
        for (uint32_t hh = 0; hh < 2; ++hh) {
            if ((mask >> hh) & 1) {
                pieces = pieces && role(p, b, hh).start;
            }
        }
        if (!pieces || !((mask >> h) & 1)) {
            return std::nullopt;
        }
        return e;
    }
    return std::nullopt;
}

std::vector<std::pair<RsPlan::Op, uint32_t>> RsPlan::ops(uint32_t p, uint32_t cx, uint32_t cy) const {
    uint32_t h = half(k(cx, cy));
    const auto& ents = entries(p, h);
    const auto passes = column_passes(p, cx);
    std::map<uint32_t, uint32_t> last;
    for (uint32_t i = 0; i < passes.size(); ++i) {
        if (passes[i].entry) {
            last[*passes[i].entry] = i;
        } else {
            for (uint32_t r = 0; r < MT; ++r) {
                last[index_of(ents, (cy * MT + r) / RB)] = i;
            }
        }
    }
    std::vector<uint32_t> pcs;
    for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
        auto e = piece_entry(p, h, d);
        if (e && count(cx, cy, ents[*e])) {
            pcs.push_back(*e);
        }
    }
    std::vector<std::pair<Op, uint32_t>> out;
    std::vector<uint32_t> pending;
    for (uint32_t i = 0; i < passes.size(); ++i) {
        out.emplace_back(Op::Pass, i);
        for (uint32_t e : pcs) {
            out.emplace_back(Op::Hand, e);
        }
        std::vector<uint32_t> done;
        for (const auto& [e, li] : last) {
            if (li == i) {
                done.push_back(e);  // (std::map: sorted)
            }
        }
        if (!pending.empty() && !done.empty()) {
            for (uint32_t e : pending) {
                out.emplace_back(Op::Fin, e);
            }
            pending.clear();
        }
        for (uint32_t e : done) {
            uint32_t b = ents[e];
            if (count(cx, cy, b) == 0 || std::find(pcs.begin(), pcs.end(), e) != pcs.end()) {
                continue;
            }
            const auto& r = role(p, b, h);
            if (r.dir != RsDir::Fin && r.start) {
                out.emplace_back(Op::Hand, e);
            } else {
                pending.push_back(e);
            }
        }
    }
    for (uint32_t e : pending) {
        out.emplace_back(Op::Fin, e);
    }
    return out;
}

std::vector<std::pair<uint32_t, RsDir>> RsPlan::handoffs(uint32_t p, uint32_t cx, uint32_t cy) const {
    uint32_t h = half(k(cx, cy));
    const auto& ents = entries(p, h);
    std::vector<std::pair<uint32_t, RsDir>> out;
    for (const auto& [kind_, e] : ops(p, cx, cy)) {
        if (kind_ == Op::Pass) {
            continue;
        }
        RsDir d = role(p, ents[e], h).dir;
        if (kind_ == Op::Hand || (kind_ == Op::Fin && d != RsDir::Fin)) {
            out.emplace_back(e, d);
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> RsPlan::zone_slots(uint32_t p, uint32_t cx, uint32_t cy, RsDir d) const {
    const auto& arr = transport.chips[p].arrivals;
    auto it = arr.find({half(k(cx, cy)), d});
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (it == arr.end()) {
        return out;
    }
    const auto& lst = it->second;
    if (!compact) {
        for (uint32_t s_ = 0; s_ < lst.size(); ++s_) {
            out.emplace_back(s_ * P, P);
        }
        return out;
    }
    uint32_t o = 0;
    for (uint32_t b : lst) {
        uint32_t n = count(cx, cy, b);
        out.emplace_back(o, n);
        o += n;
    }
    return out;
}

uint32_t RsPlan::zone_tiles(RsDir d) const {
    if (!compact) {
        return transport.slots[static_cast<int>(d)] * P;
    }
    uint32_t m = 0;
    for (uint32_t p = 0; p < G; ++p) {
        for (uint32_t cy = 0; cy < GY; ++cy) {
            for (uint32_t cx = 0; cx < GX; ++cx) {
                uint32_t n = 0;
                for (const auto& [_, t] : zone_slots(p, cx, cy, d)) {
                    n += t;
                }
                m = std::max(m, n);
            }
        }
    }
    return m;
}

std::optional<uint32_t> RsPlan::up(uint32_t p, RsDir d) const {
    int64_t q = d == RsDir::Fwd ? int64_t(p) - 1 : int64_t(p) + 1;
    if (kind != RsKind::Line) {
        return uint32_t(pymod(q, G));
    }
    if (q >= 0 && q < G) {
        return uint32_t(q);
    }
    return std::nullopt;
}

std::array<std::vector<RsPlan::Arrival>, 2> RsPlan::arrivals(uint32_t p, uint32_t cx, uint32_t cy) const {
    const uint32_t h = half(k(cx, cy));
    const auto& sc = transport.chips[p];
    const auto& ents = sc.entries[h];
    std::array<std::vector<Arrival>, 2> out;
    for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
        const auto& lst = sc.arrivals.at({h, d});
        if (lst.empty()) {
            continue;
        }
        const uint32_t u = *up(p, d);
        std::vector<uint32_t> seq;
        for (uint32_t b : transport.chips[u].entries[h]) {
            if (transport.chips[u].roles.at({b, h}).dir == d && count(cx, cy, b) > 0) {
                seq.push_back(b);
            }
        }
        for (uint32_t slot = 0; slot < lst.size(); ++slot) {
            const uint32_t b = lst[slot];
            out[static_cast<int>(d)].push_back(
                {index_of(ents, b), slot, count(cx, cy, b) > 0 ? index_of(seq, b) + 1 : 0});
        }
    }
    return out;
}

std::vector<RsPlan::PortUnit> RsPlan::port_units(uint32_t p, RsDir d) const {
    const auto& sc = transport.chips[p];
    const auto q = down(p, d);
    std::vector<PortUnit> out;
    const auto& ents = sc.entries[0];
    for (uint32_t e = 0; e < ents.size(); ++e) {
        const uint32_t b = ents[e];
        uint32_t mask = 0;
        for (uint32_t h = 0; h < 2; ++h) {
            if (role(p, b, h).dir == d) {
                mask |= 1u << h;
            }
        }
        if (!mask) {
            continue;
        }
        uint32_t slots[2] = {0, 0};
        for (uint32_t h = 0; h < 2; ++h) {
            if ((mask >> h) & 1) {
                slots[h] = index_of(transport.chips[*q].arrivals.at({h, d}), b);
            }
        }
        bool pieces = pm && out.empty();
        for (uint32_t h = 0; h < 2; ++h) {
            if ((mask >> h) & 1) {
                pieces = pieces && role(p, b, h).start;
            }
        }
        out.push_back({e, b, mask, slots[0], slots[1], pieces});
    }
    return out;
}

std::vector<RsDir> RsPlan::finb_readers(uint32_t p, uint32_t cx, uint32_t cy) const {
    const uint32_t h = half(k(cx, cy));
    std::vector<RsDir> out;
    for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
        if (!down(p, d)) {
            continue;
        }
        for (const auto& u : port_units(p, d)) {
            if (((u.mask >> h) & 1) && count(cx, cy, u.block) > 0) {
                out.push_back(d);
                break;
            }
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> RsPlan::port_cores(uint32_t l) const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t cy = 0; cy < GY; ++cy) {
        for (uint32_t cx = 0; cx < GX; ++cx) {
            if (link(k(cx, cy)) == l) {
                out.emplace_back(cx, cy);
            }
        }
    }
    return out;
}

std::tuple<uint32_t, uint32_t, uint32_t> RsPlan::fin_map(uint32_t p, uint32_t cx, uint32_t cy) const {
    if (scatter == -1) {
        return {cy * MT * NB + s[cx], NB, w[cx]};
    }
    const auto [r0, nr] = rows_in(cy, p);
    if (nr == 0) {
        return {0, NTH, 1};
    }
    return {(cy * MT + r0 - p * RB) * NTH + n0[cx], NTH, pn[cx]};
}

uint32_t RsPlan::prefix_mask(uint32_t q, uint32_t h, RsDir d, uint32_t slot) const {
    const auto& arr = transport.chips[q].arrivals;
    auto it = arr.find({h, d});
    uint32_t m = 0;
    if (it != arr.end()) {
        for (uint32_t i = 0; i < slot && i < it->second.size(); ++i) {
            m |= 1u << it->second[i];
        }
    }
    return m;
}

uint32_t RsPlan::counts_word(uint32_t cx, uint32_t cy) const {
    if (P >= 256) {
        fail("se_rs: a core's sub-block of " + std::to_string(P) + " tiles > 255 (the counts word's bytes)");
    }
    if (G <= 4) {
        uint32_t w_ = 0;
        for (uint32_t b = 0; b < G; ++b) {
            w_ += count(cx, cy, b) << (8 * b);
        }
        return w_;
    }
    std::vector<std::pair<uint32_t, uint32_t>> nz;
    for (uint32_t b = 0; b < G; ++b) {
        if (count(cx, cy, b)) {
            nz.emplace_back(b, count(cx, cy, b));
        }
    }
    if (!(scatter == -2 && nz.size() <= 2)) {
        fail("se_rs: TP > 4 needs the rows scatter with a core's rows in at most two blocks");
    }
    while (nz.size() < 2) {
        nz.emplace_back(0xFF, 0);
    }
    return nz[0].first | nz[0].second << 8 | nz[1].first << 16 | nz[1].second << 24;
}

uint32_t RsPlan::finb_off(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const {
    const auto& ents = entries(p, half(k(cx, cy)));
    uint32_t n = 0;
    for (uint32_t i = 0; i < e; ++i) {
        n += count(cx, cy, ents[i]);
    }
    return n;
}

std::vector<uint32_t> RsPlan::pass_widths(uint32_t cx) const {
    std::vector<uint32_t> out;
    for (const auto& ps : column_passes(0, cx)) {
        out.push_back(ps.pw);
    }
    return out;
}

std::vector<std::pair<uint32_t, std::vector<uint32_t>>> RsPlan::pass_bases(uint32_t p, uint32_t cx, uint32_t cy) const {
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> out;
    const auto& ents = entries(p, half(k(cx, cy)));
    const auto hold = hold_offs(p, cx, cy).first;
    auto at = [&](uint32_t e, uint32_t within) -> uint32_t {
        auto it = hold.find(e);
        if (it != hold.end()) {
            return (it->second + within) | (1u << 31);
        }
        return finb_off(p, cx, cy, e) + within;
    };
    for (const auto& ps : column_passes(p, cx)) {
        std::vector<uint32_t> bases;
        if (scatter == -1) {
            const uint32_t e = *ps.entry, b = ents[e];
            const uint32_t off = ps.c0 - b * NB - s[cx];
            for (uint32_t r = 0; r < MT; ++r) {
                bases.push_back(at(e, r * w[cx] + off));
            }
        } else {
            for (uint32_t r = 0; r < MT; ++r) {
                const uint32_t b = (cy * MT + r) / RB;
                const auto [r0, nr] = rows_in(cy, b);
                if (pm) {
                    bases.push_back(at(index_of(ents, b), nr * (ps.c0 - n0[cx]) + (r - r0) * ps.pw));
                } else {
                    bases.push_back(at(index_of(ents, b), (r - r0) * pn[cx] + ps.c0 - n0[cx]));
                }
            }
        }
        out.emplace_back(ps.pw, bases);
    }
    return out;
}

std::vector<uint32_t> RsPlan::compute_args(uint32_t p, uint32_t cx, uint32_t cy) const {
    const auto o = ops(p, cx, cy);
    std::vector<uint32_t> out = {uint32_t(o.size())};
    for (const auto& [kind_, i] : o) {
        out.push_back(static_cast<uint32_t>(kind_) | (i << 8));
    }
    for (const auto& [pw, bases] : pass_bases(p, cx, cy)) {
        out.push_back(pw);
        out.insert(out.end(), bases.begin(), bases.end());
    }
    const uint32_t h = half(k(cx, cy));
    const auto& ents = entries(p, h);
    const auto hold = hold_offs(p, cx, cy).first;
    const auto arr = arrivals(p, cx, cy);
    const auto zf = zone_slots(p, cx, cy, RsDir::Fwd), zb = zone_slots(p, cx, cy, RsDir::Bwd);
    for (uint32_t e = 0; e < ents.size(); ++e) {
        const uint32_t b = ents[e];
        const RsDir d = role(p, b, h).dir;
        std::optional<uint32_t> sf, sb;
        for (const auto& a : arr[0]) {
            if (a.entry == e && !sf) {
                sf = a.slot;
            }
        }
        for (const auto& a : arr[1]) {
            if (a.entry == e && !sb) {
                sb = a.slot;
            }
        }
        const uint32_t mode = (sf ? 1 : 0) | (sb ? 2 : 0) | (d == RsDir::Fin ? 4 : 0);
        auto it = hold.find(e);
        out.push_back(count(cx, cy, b));
        out.push_back(mode);
        out.push_back(sf ? zf[*sf].first : 0);
        out.push_back(sb ? zb[*sb].first : 0);
        out.push_back(it != hold.end() ? it->second : 0);
        out.push_back(finb_off(p, cx, cy, e));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// SeEngine3 sizing

SeEngineSizing::SeEngineSizing(const SeEngineArgs& a) : args(a) {
    GX = a.GX;
    GY = a.GY;
    MT = a.MT;
    CR = GY * MT;
    rows = a.rows;
    H = a.H;
    I = a.I;
    shards = a.shards;
    if (a.rows % (32 * CR) != 0) {
        fail(
            "se_engine3: " + std::to_string(a.rows) + " rows are not whole chunks of CR = GY " + std::to_string(GY) +
            " x MT " + std::to_string(MT) + " = " + std::to_string(CR) + " row tiles");
    }
    NCH = a.rows / 32 / CR;
    KT = a.H / 32;
    HT = a.I / 32;
    NTH = a.H / 32;
    KB = a.KB;
    NSG = KT / KB;
    if (!(KT % KB == 0 && NSG % 2 == 0)) {
        fail(
            "se_engine3: K step " + std::to_string(KB) + " must divide H's " + std::to_string(KT) +
            " tiles into an even count of steps");
    }
    const int64_t arena_max = a.arena_max;
    if (!(NSG % a.shards == 0 && (NSG / a.shards) % 2 == 0)) {
        fail("even K steps a shard");
    }
    SPS = NSG / a.shards;
    const uint32_t nf = NSG / 2;
    std::tie(S0, S1) = subsets(SPS / 2, GX);
    for (uint32_t c = 0; c < S0; ++c) {
        fam0.push_back(c);
    }
    for (uint32_t c = GX - S1; c < GX; ++c) {
        fam1.push_back(c);
    }
    const auto pairs = column_split(HT, GX, 1);
    if (vmin(pairs) < 1) {
        fail(
            "se_engine3: I's " + std::to_string(HT) + " tiles leave a column of the " + std::to_string(GX) +
            " without a gate / up pair");
    }
    po = pairs;
    h0 = prefix(pairs);
    POMAX = vmax(pairs);
    const uint32_t DT = kDestTiles;
    PGW = 8 / MT;
    const uint32_t pwmax = PGW / 2 * 2;
    NPG = 0;
    for (uint32_t p : pairs) {
        NPG = std::max<uint32_t>(NPG, ceil_div(2 * p, pwmax));
    }
    for (uint32_t p : pairs) {
        wg.push_back(split_pad(2 * p, NPG, 2));
    }
    NGU = NPG;
    LT = a.latent.value_or(0);
    if (a.latent) {
        lw = column_split(*a.latent, GX, 1);
        l0 = prefix(lw);
        if (vmin(lw) < 1) {
            fail(
                "se_engine3: the latent's " + std::to_string(*a.latent) + " tiles leave a column of the " +
                std::to_string(GX) + " without one");
        }
        LWMAX = vmax(lw);
        const uint32_t npl = ceil_div(LWMAX, PGW);
        for (uint32_t c = 0; c < GX; ++c) {
            auto extra = split_units(lw[c], npl, 1);
            wg[c].insert(wg[c].end(), extra.begin(), extra.end());
        }
        NPG += npl;
    }
    LATP = a.latent ? int64_t(NCH > 1 ? a.shards : 1) * MT * LWMAX : 0;
    const auto pnd = column_split(NTH, GX, 1);
    n0d = prefix(pnd);
    PWDMAX = DT / MT;
    NPD = 0;
    for (uint32_t p : pnd) {
        NPD = std::max<uint32_t>(NPD, ceil_div(p, PWDMAX));
    }
    for (uint32_t p : pnd) {
        wd.push_back(split_units(p, NPD, 1));
    }
    const RsPlan* rs = a.rs;
    if (rs == nullptr) {
        fail("se_engine3: no reduce-scatter plan");
    }
    {
        if (!(rs->pwmax == PWDMAX && rs->MT == MT && rs->GX == GX && rs->GY == GY)) {
            fail(
                "se_engine3: the RS plan's tiling (GX " + std::to_string(rs->GX) + " GY " + std::to_string(rs->GY) +
                " MT " + std::to_string(rs->MT) + " pwmax " + std::to_string(rs->pwmax) +
                ") is not the matmul cores' (GX " + std::to_string(GX) + " GY " + std::to_string(GY) + " MT " +
                std::to_string(MT) + " pwmax " + std::to_string(PWDMAX) + ")");
        }
        wd.clear();
        for (uint32_t c = 0; c < GX; ++c) {
            std::vector<uint32_t> ws;
            for (const auto& ps : rs->column_passes(0, c)) {
                ws.push_back(ps.pw);
            }
            wd.push_back(ws);
        }
        NPD = wd[0].size();
        for (const auto& x : wd) {
            if (x.size() != NPD) {
                fail("se_engine3: uneven down passes " + nested_str(wd));
            }
        }
    }
    KBD = 1;
    for (uint32_t d = 1; d <= 8; ++d) {
        if (HT % d == 0) {
            KBD = d;
        }
    }
    NSD = HT / KBD;
    NB = NPG * NSG + NPD * NSD;
    if (NB > 1024) {
        fail(
            "se_engine3: " + std::to_string(NB) + " W rounds a chunk > 1024 (NPG " + std::to_string(NPG) + " x NSG " +
            std::to_string(NSG) + " + NPD " + std::to_string(NPD) + " x NSD " + std::to_string(NSD) +
            "): a larger K step than " + std::to_string(KB));
    }
    uint32_t wgmax = 0, wdmax = 0;
    for (const auto& x : wg) {
        wgmax = std::max(wgmax, vmax(x));
    }
    for (const auto& x : wd) {
        wdmax = std::max(wdmax, vmax(x));
    }
    WSLOT = std::max(KB * wgmax, KBD * wdmax);
    switch (a.w_dtype) {
        case WeightDtype::Bf8: wtb = 1088; break;
        case WeightDtype::Bf4: wtb = 576; break;
        case WeightDtype::Bf16: wtb = 2048; break;
    }
    wown_t = 0;
    for (uint32_t c = 0; c < GX; ++c) {
        for (uint32_t cy = 0; cy < GY; ++cy) {
            int64_t sum = 0;
            for (uint32_t b = cy; b < NB; b += GY) {
                sum += bsize(c, b);
            }
            wown_t = std::max(wown_t, sum);
        }
    }
    uint32_t nown = 0;
    for (uint32_t c = 0; c < GX; ++c) {
        nown = std::max(nown, (in_fam0(c) ? nf / S0 : 0) + (in_fam1(c) ? nf / S1 : 0));
    }
    XR = a.xring ? std::min(a.xring, NCH) : NCH;
    xring = XR < NCH;
    xfree = xring && !a.latent;
    xown_t = int64_t(XR) * nown * MT * KB;
    PFULL = a.latent ? LWMAX : KB;
    for (uint32_t d = 1; d <= kDestTiles; ++d) {
        if (PFULL % d == 0) {
            PBCT = d;
        }
    }

    // se_engine3._rs_regions: the same at every W ring / h slot count
    Regions rsr0;
    {
        uint32_t hold = 0, finb = 0;
        size_t nhand = 0;
        for (uint32_t p = 0; p < rs->G; ++p) {
            for (uint32_t cx = 0; cx < GX; ++cx) {
                for (uint32_t cy = 0; cy < GY; ++cy) {
                    hold = std::max(hold, rs->hold_offs(p, cx, cy).second);
                    nhand = std::max(nhand, rs->handoffs(p, cx, cy).size());
                }
            }
        }
        for (uint32_t cx = 0; cx < GX; ++cx) {
            for (uint32_t cy = 0; cy < GY; ++cy) {
                finb = std::max(finb, rs->finb_tiles(cx, cy));
            }
        }
        rsr0 = {
            {"hold", int64_t(std::max(1u, hold)) * 4096},
            {"finb", int64_t(std::max(1u, finb)) * 2048},
            {"arrf", int64_t(rs->zone_tiles(RsDir::Fwd)) * 2048},
            {"arrb", int64_t(rs->zone_tiles(RsDir::Bwd)) * 2048},
            {"outf", 8 * 2048},
            {"finc", 16 * int64_t(std::max<size_t>(8, nhand))},
            {"psync", 16 * int64_t(std::max<uint32_t>(8, NPD))},
        };
        if (NCH > 1) {
            rsr0.emplace_back("ffree", 16 * int64_t(std::max<uint32_t>(2, NCH - 1)));
        }
    }
    rs_sizes = rsr0;
    const std::vector<std::pair<uint32_t, uint32_t>> tries =
        NCH == 1 ? std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {5, 1}, {4, 1}, {3, 1}}
                 : std::vector<std::pair<uint32_t, uint32_t>>{{0, 2}, {5, 2}, {5, 1}, {4, 1}, {3, 1}};
    int64_t o = 0;
    for (const auto& [wring, hsl] : tries) {
        WRING = wring;
        HSL = hsl;
        uint32_t busiest = 0;
        for (const auto& x : wg) {
            busiest = std::max(busiest, vsum(x));
        }
        acc_pages = int64_t(MT) * busiest;
        off.clear();
        o = 0;
        Regions rsr = rsr0;
        Regions ovl, ovh;
        if (!rsr.empty() && NCH == 1) {
            ovl.assign(rsr.begin(), rsr.begin() + 2);
            rsr.erase(rsr.begin(), rsr.begin() + 2);
        }
        if (!rsr.empty() && NCH > 1) {
            ovh.assign(rsr.begin(), rsr.begin() + 1);
            rsr.erase(rsr.begin(), rsr.begin() + 1);
        }
        Regions seq = {
            {"xown", xown_t * 2048},
            {"wown", (WRING ? int64_t(WRING) * WSLOT : wown_t) * wtb},
            {"win", int64_t(3) * WSLOT * wtb},  // (NWS W slots)
            {"hown", int64_t(MT) * POMAX * 2048},
            {"hall", int64_t(HSL) * GX * MT * POMAX * 2048},
            {"xin", int64_t(2) * MT * KB * 2048},
            {"acc", acc_pages * 4096},
            {"words", 256},
            {"xfree", xfree ? 16 * int64_t(NCH) : 0},
        };
        seq.insert(seq.end(), rsr.begin(), rsr.end());
        if (a.latent) {
            seq.emplace_back("lat", int64_t(MT) * LWMAX * 2048);
            seq.emplace_back("latz", int64_t(a.shards - 1) * MT * LWMAX * 2048 + int64_t(NCH - 1) * LATP * 2048);
        }
        seq.emplace_back("po", int64_t(2) * PFULL * 2048);
        if (a.latent) {
            seq.emplace_back("lrdy", 16 * int64_t(std::max<uint32_t>(4, NCH)));
        }
        for (const auto& [k_, v] : seq) {
            set_region(off, k_, o);
            o = al64(o + v);
        }
        {
            std::set<int64_t> e{o};
            for (const auto& [_, v] : off) {
                e.insert(v);
            }
            seq_ends.assign(e.begin(), e.end());
        }
        if (!ovl.empty()) {
            // (the room from xin up to the words, the region after acc)
            const int64_t room = region(off, "words") - region(off, "xin");
            const int64_t hold = region(ovl, "hold"), finb = region(ovl, "finb");
            if (al64(hold) + finb <= room) {
                set_region(off, "hold", region(off, "xin"));
                set_region(off, "finb", region(off, "xin") + al64(hold));
            } else {
                for (const auto& [k_, v] : ovl) {
                    set_region(off, k_, o);
                    o = al64(o + v);
                }
            }
        }
        if (!ovh.empty()) {
            const int64_t hold = region(ovh, "hold");
            if (hold <= acc_pages * 4096) {
                set_region(off, "hold", region(off, "acc"));
            } else {
                set_region(off, "hold", o);
                o = al64(o + hold);
            }
        }
        {  // the router logits on the engine
            const auto [ET, KTO] = a.router;
            ETC = ceil_div(ET, GX);
            const uint32_t MTR = MT * NCH;
            RG = ETC * MTR <= DT ? MTR : (ETC * MT <= DT ? MT : 1);
            if (ETC * RG > DT) {
                fail("the router logits on the engine: ETC x its row group in DEST");
            }
            int64_t zgrow = 0;
            if (rs->compact) {
                for (RsDir d : {RsDir::Fwd, RsDir::Bwd}) {
                    zgrow += (int64_t(rs->transport.slots[static_cast<int>(d)]) * rs->P - rs->zone_tiles(d)) * 2048;
                }
            }
            int64_t t0 = rsr.empty() ? o : region(off, rsr[0].first);
            std::vector<uint32_t> kcs;
            for (uint32_t d = KTO; d >= 1; --d) {
                if (KTO % d == 0 && KTO / d >= 2) {
                    kcs.push_back(d);
                }
            }
            if (kcs.empty()) {
                fail("se_engine3: no router K chunk");
            }
            Regions rsz;
            int64_t need = 0;
            const int64_t lim = std::min(o + zgrow, std::max(o, arena_max));
            for (uint32_t kc : kcs) {
                KC = kc;
                rsz = {
                    {"rx", int64_t(2) * KC * MTR * 2048},
                    {"rw", int64_t(2) * ETC * KC * 2048},
                    {"ro", int64_t(ETC) * MTR * 4096},
                };
                need = 0;
                for (const auto& [_, v] : rsz) {
                    need += al64(v);
                }
                if (t0 + need <= lim || KC == kcs.back() ||
                    (KC == kcs.front() && KTO / KC == 2 && o + need <= arena_max)) {
                    break;
                }
            }
            if (o - t0 < need && t0 + need <= lim) {
                o = t0 + need;
            } else if (o - t0 < need) {
                const int64_t hall = region(off, "xin") - region(off, "hall");
                t0 = hall >= need ? region(off, "hall") : o;
                if (t0 == o) {
                    o = o + need;
                }
            }
            for (const auto& [k_, v] : rsz) {
                set_region(off, k_, t0);
                t0 = al64(t0 + v);
            }
        }
        arena_bytes = o;
        if (o <= arena_max) {
            break;
        }
    }
    std::set<int64_t> ends{o};
    for (const auto& [_, v] : off) {
        ends.insert(v);
    }
    sizes.clear();
    for (const auto& [k_, v] : off) {
        sizes.emplace_back(k_, v < o ? *ends.upper_bound(v) - v : 0);
    }
    {  // (the router's buffers over a region another chip writes: those writes wait for "router done")
        const int64_t lo = at("rx"), hi = at("ro") + int64_t(ETC) * MT * NCH * 4096;
        auto over = [&](const std::string& k, int64_t v) { return v > 0 && at(k) < hi && lo < at(k) + v; };
        rtr_zones = over("arrf", region(rs_sizes, "arrf")) || over("arrb", region(rs_sizes, "arrb"));
        if (a.latent) {
            rtr_latz = over("latz", int64_t(shards - 1) * MT * LWMAX * 2048 + int64_t(NCH - 1) * LATP * 2048);
        }
    }
    rtr_overlay = rtr_zones || rtr_latz;
    if (o > arena_max) {
        throw ArenaTooLarge("se_engine3: arena too large", o, arena_max);
    }
    for (uint32_t c = 0; c < GX; ++c) {
        auto key = std::make_pair(wg[c], wd[c]);
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == key; });
        if (it == groups.end()) {
            groups.push_back({key, {c}});
        } else {
            it->second.push_back(c);
        }
    }
}

int64_t SeEngineSizing::at(const std::string& k) const { return region(off, k); }
bool SeEngineSizing::in_fam0(uint32_t c) const { return std::find(fam0.begin(), fam0.end(), c) != fam0.end(); }
bool SeEngineSizing::in_fam1(uint32_t c) const { return std::find(fam1.begin(), fam1.end(), c) != fam1.end(); }

std::tuple<uint32_t, uint32_t, uint32_t> SeEngineSizing::owner(uint32_t j) const {
    const uint32_t f = j & 1;
    const auto& fam = f ? fam1 : fam0;
    const uint32_t S = f ? S1 : S0;
    return {fam[(j / 2) % S], f, (j / 2) % S};
}

uint32_t SeEngineSizing::grow(uint32_t m, uint32_t lr) const {
    if (NCH == 1 || shards == 1) {
        return m * CR + lr;
    }
    const uint32_t rbc = CR / shards;
    return (lr / rbc) * (rows / 32 / shards) + m * rbc + lr % rbc;
}

std::pair<uint32_t, uint32_t> SeEngineSizing::la_rows(uint32_t cy, uint32_t b) const {
    const uint32_t RB = CR / shards;
    uint32_t first = 0, n = 0;
    for (uint32_t r = 0; r < MT; ++r) {
        if ((cy * MT + r) / RB == b) {
            first = n ? first : r;
            ++n;
        }
    }
    return {n ? first : 0, n};
}

uint32_t SeEngineSizing::bsize(uint32_t c, uint32_t b) const {
    return b < NPG * NSG ? KB * wg[c][b / NSG] : KBD * wd[c][(b - NPG * NSG) / NSD];
}

bool SeEngineSizing::fin_rows() const {
    const RsPlan* rs = args.rs;
    return rs->scatter == -1 && NCH > 1 && shards > 1 && (CR / shards) % MT != 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// RouterStream / routing_a2a

std::pair<Regions, int64_t> router_layout(uint32_t G, uint32_t ET, uint32_t K, uint32_t ETF) {
    ETF = ETF ? ETF : ET;
    Regions spec = {
        {"hold", int64_t(G) * ET * 4096},
        {"sc", int64_t(ET) * 4096},
        {"idx", int64_t(ET) * 2048},
        {"best", 4096},
        {"bias", int64_t(ETF) * 4096},
        {"arr", int64_t(G - 1) * ET * 4096},
        {"finc", 16 * 8},
        {"psync", 16 * 2},
        {"arrb", 16 * int64_t(std::max<uint32_t>(1, G - 1))},
        {"rdy", 16 * 2},
        {"out", 32 * 4 * int64_t(K)},
        {"words", 160},
        {"route", 32 * 4 * 2},
        {"msync", 16},
    };
    if (ETF != ET) {
        spec.emplace_back("part", 2 * 4096);
    }
    Regions off;
    int64_t o = 0;
    for (const auto& [k_, v] : spec) {
        off.emplace_back(k_, o);
        o = al64(o + v);
    }
    return {off, o};
}

int64_t router_min_bytes(uint32_t G, uint32_t E, uint32_t K) {
    const uint32_t ET = E / 32;
    return router_layout(G, ET, K, ET).second;
}

int64_t routing_header_bytes(uint32_t EPC, uint32_t M) {
    return al64(64 + al16(2 * int64_t(EPC)) + int64_t(M) * al16(2 * (int64_t(EPC) + 1)));
}

int64_t routing_slot_bytes(uint32_t S, uint32_t K, uint32_t EPC, uint32_t M) {
    return al64(routing_header_bytes(EPC, M) + 4 * int64_t(S) * std::min(K, EPC));
}

int64_t routing_worst_rows(uint32_t N, uint32_t S, uint32_t K, uint32_t EPC) {
    return (int64_t(N) * S * std::min(K, EPC) + 31) / 32 * 32 + 32 * int64_t(EPC);
}

RoutingSizes routing_a2a_sizes(
    uint32_t N, uint32_t S, uint32_t K, uint32_t EPC, uint32_t A, int64_t rows_cap, uint32_t M, uint32_t TPP) {
    M = M ? M : std::min({8u, EPC, A});
    const int64_t NG = int64_t(N) * EPC;
    const int64_t HBP = routing_header_bytes(EPC, M);
    const int64_t PCAP = routing_slot_bytes(S, K, EPC, M);
    const int64_t NOWN = (A + M - 1) / M;
    const int64_t TSM = NOWN * ((S + A - 1) / A);
    const int64_t BBP = al16(2 * (int64_t(EPC) + 1));
    const int64_t mk = std::min(K, EPC);
    const int64_t inst_cap = (ceil_div(int64_t(N) * S * mk, M) + int64_t(N) * S + 31) / 32 * 32;
    const int64_t IC = std::min(inst_cap, rows_cap);
    const int64_t IR = std::min(IC + 32 * int64_t(EPC), rows_cap);
    const int64_t NM = (int64_t(N) + M - 1) / M;
    const int64_t plan_l1 = al64((S / TPP) * al64(int64_t(TPP) * 4 * K)) + al64(4 * int64_t(M) * N) + al64(4 * N) +
                            N * BBP + al64(4 * TSM * K + 16 * int64_t(N)) + N * HBP +
                            al64(4 * IC + 128 * int64_t(N) * M) + al64(4 * int64_t(N) * M) +
                            al64(2 * int64_t(EPC) * S) + al64(4 * IR) + al64(8 * IR) + al64(128 * NOWN) + M * 64 + 128 +
                            2 * al64(4 * (int64_t(EPC) + 32)) + 64 + N * al16(2 * int64_t(EPC)) +
                            NM * M * al16(2 * int64_t(EPC)) + al64(NM * al16(2 * int64_t(EPC))) + 128;
    const int64_t port_l1 = int64_t(N) * PCAP + 64;
    std::vector<std::string> why;
    if (N > 32 || A > 16 || K % 2 || S % TPP || !(1 <= M && M <= std::min({EPC, A, 16u}))) {
        why.push_back(
            "shape: N " + std::to_string(N) + " <= 32, A " + std::to_string(A) + " <= 16, K " + std::to_string(K) +
            " even, S " + std::to_string(S) + " % " + std::to_string(TPP) + ", M " + std::to_string(M) +
            " <= min(EPC, A, 16)");
    }
    if (NG > 1024) {
        why.push_back("local counters: N EPC " + std::to_string(NG) + " > 1024");
    }
    if (S > 2048 || EPC > 64) {
        why.push_back("an entry: S " + std::to_string(S) + " <= 2048 and EPC " + std::to_string(EPC) + " <= 64");
    }
    const int64_t l1 = 1'400'000;
    if (plan_l1 > l1 || port_l1 > l1) {
        why.push_back(
            "L1: planner " + std::to_string(plan_l1 >> 10) + " KB, port " + std::to_string(port_l1 >> 10) + " KB");
    }
    std::string w;
    for (size_t i = 0; i < why.size(); ++i) {
        w += (i ? "; " : "") + why[i];
    }
    return RoutingSizes{M, IC, plan_l1, port_l1, why.empty(), w};
}

// ---------------------------------------------------------------------------------------------------------------------
// routing_paths: head_plan.host_chips / routing_shell + RoutingStream.serve / path / fwd_plan

namespace head_sizing_detail {

enum : int { kFwd = 1, kBwd = 0 };  // (x sender keys sort "bwd" < "fwd")

// A sender key, ordered as the reference's tuples: ("x", "bwd" | "fwd", link) < ("y", 2) < ("y", 3).
using SenderKey = std::tuple<int, int, int>;  // (0 x | 1 y, x: direction / y: 2 | 3, link)

struct HostChip {
    std::optional<std::pair<int, int>> next, prev;
    bool sends[2] = {false, false};  // [kBwd], [kFwd]: the gather's schedule toward prev / next is non-empty
    bool links[2] = {false, false};  // a direction carries links
};

std::string coord_str(int r, int c) { return "(" + std::to_string(r) + ", " + std::to_string(c) + ")"; }

}  // namespace head_sizing_detail

std::variant<RoutingPaths, std::string> routing_paths(uint32_t R, uint32_t C, bool ring, bool wrap_y, uint32_t links) {
    const int L = 2;  // (ROUTING_L: the routing exchange's port tables)
    const int N = int(R * C);
    std::map<std::pair<int, int>, HostChip> chips;
    for (int r = 0; r < int(R); ++r) {
        for (int p = 0; p < int(C); ++p) {
            HostChip ch;
            if (ring || p < int(C) - 1) {
                ch.next = std::make_pair(r, int(pymod(p + 1, C)));
            }
            if (ring || p > 0) {
                ch.prev = std::make_pair(r, int(pymod(p - 1, C)));
            }
            // the gather's schedule: which directions send shards
            int G = int(C);
            if (ring && G % 2 == 0 && G >= 4) {
                ch.sends[kFwd] = G / 2 > 0;
                ch.sends[kBwd] = G / 2 > 0;
            } else if (ring) {
                ch.sends[kFwd] = G / 2 > 0;
                ch.sends[kBwd] = G - 1 - G / 2 > 0;
            } else {
                ch.sends[kFwd] = p < G - 1;
                ch.sends[kBwd] = p > 0;
            }
            chips[{r, p}] = ch;
        }
    }
    for (auto& [coord, ch] : chips) {
        for (int d : {kFwd, kBwd}) {
            const auto& peer = d == kFwd ? ch.next : ch.prev;
            // (_needs_ready: the peer's opposite-direction port sends to this chip)
            if (peer && (ch.sends[d] || chips.at(*peer).sends[1 - d])) {
                ch.links[d] = links > 0;
            }
        }
    }
    auto first_hop = [&](int s, int d) {
        int sr = s / int(C), sc = s % int(C), dr = d / int(C), dc = d % int(C);
        if (sr != dr) {
            if (wrap_y) {
                return 2 * pymod(dr - sr, R) < int(R) ? 2 : 3;
            }
            return dr > sr ? 2 : 3;
        }
        if (ring) {
            return 2 * pymod(dc - sc, C) <= int(C) ? 0 : 1;
        }
        return dc > sc ? 0 : 1;
    };
    auto xdir = [&](int r, int c) {
        std::map<int, int> out;  // 0 (X+) | 1 (X-) -> the gather's direction toward it
        const auto& rg = chips.at({r, c});
        for (int d : {kFwd, kBwd}) {
            const auto& peer = d == kFwd ? rg.next : rg.prev;
            if (peer && rg.links[d]) {
                bool fwd = ring ? pymod(peer->second - c, C) == 1 : peer->second > c;
                out[fwd ? 0 : 1] = d;
            }
        }
        return out;
    };
    // [(chip, key)] of the packet s -> d
    auto path = [&](int s, int d) {
        std::vector<std::pair<int, SenderKey>> steps;
        int sr = s / int(C), sc = s % int(C), dr = d / int(C), dc = d % int(C);
        std::pair<int, int> cur{sr, sc};
        if (sr != dr) {
            steps.emplace_back(s, SenderKey{1, first_hop(s, d), 0});
            cur = {dr, sc};
        }
        while (cur.second != dc) {
            int ci = cur.first * int(C) + cur.second;
            int h = first_hop(ci, d);
            auto xd = xdir(cur.first, cur.second);
            if (!xd.count(h)) {
                throw SizingError(
                    "chip " + coord_str(cur.first, cur.second) + ": no X port toward " + std::to_string(d));
            }
            int l = (d % L) < int(links) ? d % L : 0;
            steps.emplace_back(ci, SenderKey{0, xd[h], l});
            cur = {cur.first, int(pymod(cur.second + (h == 0 ? 1 : -1), C))};
        }
        return steps;
    };
    try {
        uint32_t nx = 0, ny = 0, nsend = 0;
        for (const auto& [coord, _] : chips) {
            int me = coord.first * int(C) + coord.second;
            std::map<SenderKey, uint32_t> out;
            for (int dd = 0; dd < N; ++dd) {
                if (dd != me) {
                    ++out[path(me, dd)[0].second];
                }
            }
            std::vector<SenderKey> ykeys;
            for (const auto& [k_, n] : out) {
                if (std::get<0>(k_) == 1) {
                    ykeys.push_back(k_);
                }
            }
            ++out[ykeys.empty() ? out.begin()->first : ykeys[0]];  // (my own chip)
            if (ykeys.size() > 2) {
                throw SizingError(
                    "chip " + coord_str(coord.first, coord.second) + ": " + std::to_string(ykeys.size()) +
                    " Y directions");
            }
            for (const auto& [k_, n] : out) {
                (std::get<0>(k_) == 0 ? nx : ny) = std::max(std::get<0>(k_) == 0 ? nx : ny, n);
            }
            nsend = std::max<uint32_t>(nsend, out.size());
        }
        std::map<std::pair<int, SenderKey>, uint32_t> fw;
        for (int s_ = 0; s_ < N; ++s_) {
            for (int d = 0; d < N; ++d) {
                if (d == s_) {
                    continue;
                }
                auto steps = path(s_, d);
                for (size_t i = 1; i < steps.size(); ++i) {
                    ++fw[steps[i]];
                }
            }
        }
        uint32_t nf = 0;
        for (const auto& [_, n] : fw) {
            nf = std::max(nf, n);
        }
        return RoutingPaths{nx, ny, nf, nsend};
    } catch (const SizingError& e) {
        return std::string(e.what());
    }
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head
