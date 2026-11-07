// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_routing.hpp"

#include <algorithm>
#include <tuple>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace routing_detail {

// Python's a % n for possibly negative a.
int64_t pmod(int64_t a, int64_t n) { return ((a % n) + n) % n; }

}  // namespace routing_detail

std::string_view dir_name(Dir d) {
    switch (d) {
        case Dir::XPlus: return "X+";
        case Dir::XMinus: return "X-";
        case Dir::YPlus: return "Y+";
        case Dir::YMinus: return "Y-";
    }
    return "?";
}

Dir opposite(Dir d) {
    switch (d) {
        case Dir::XPlus: return Dir::XMinus;
        case Dir::XMinus: return Dir::XPlus;
        case Dir::YPlus: return Dir::YMinus;
        case Dir::YMinus: return Dir::YPlus;
    }
    return d;
}

MeshTopology MeshTopology::of(uint32_t R, uint32_t C, bool torus_x, bool torus_y) {
    return MeshTopology{.R = R, .C = C, .wrap_x = torus_x && C > 2, .wrap_y = torus_y && R > 2};
}

std::optional<ChipCoord> MeshTopology::neighbor(ChipCoord ch, Dir d) const {
    const int64_t step = is_plus(d) ? 1 : -1;
    if (is_x(d)) {
        int64_t c2 = static_cast<int64_t>(ch.c) + step;
        if (wrap_x) {
            c2 = routing_detail::pmod(c2, C);
        }
        if (c2 < 0 || c2 >= static_cast<int64_t>(C)) {
            return std::nullopt;
        }
        return ChipCoord{ch.r, static_cast<uint32_t>(c2)};
    }
    int64_t r2 = static_cast<int64_t>(ch.r) + step;
    if (wrap_y) {
        r2 = routing_detail::pmod(r2, R);
    }
    if (r2 < 0 || r2 >= static_cast<int64_t>(R)) {
        return std::nullopt;
    }
    return ChipCoord{static_cast<uint32_t>(r2), ch.c};
}

std::vector<Hop> route(ChipCoord src, ChipCoord home, const MeshTopology& topo) {
    // step(n, a, b, wrap, tie_plus) -> (sign, hops)
    auto step = [](int64_t n, int64_t a, int64_t b, bool wrap, bool tie_plus) -> std::pair<int, int64_t> {
        if (wrap) {
            const int64_t f = routing_detail::pmod(b - a, n);
            const bool plus = f * 2 < n || (f * 2 == n && tie_plus);
            return {plus ? 1 : -1, plus ? f : n - f};
        }
        return {b > a ? 1 : -1, b > a ? b - a : a - b};
    };
    std::vector<Hop> hops;
    uint32_t r = src.r, c = src.c;
    auto [sy, ky] = step(topo.R, r, home.r, topo.wrap_y, r % 2 == 1);
    for (int64_t i = 0; i < ky; ++i) {
        hops.push_back({{r, c}, sy > 0 ? Dir::YPlus : Dir::YMinus});
        r = static_cast<uint32_t>(routing_detail::pmod(static_cast<int64_t>(r) + sy, topo.R));
    }
    auto [sx, kx] = step(topo.C, c, home.c, topo.wrap_x, c % 2 == 0);
    for (int64_t i = 0; i < kx; ++i) {
        hops.push_back({{r, c}, sx > 0 ? Dir::XPlus : Dir::XMinus});
        c = static_cast<uint32_t>(routing_detail::pmod(static_cast<int64_t>(c) + sx, topo.C));
    }
    return hops;
}

std::optional<Dir> first_hop(ChipCoord src, ChipCoord dst, const MeshTopology& topo) {
    const auto hops = route(src, dst, topo);
    if (hops.empty()) {
        return std::nullopt;
    }
    return hops.front().dir;
}

uint8_t span_entry(uint32_t n, bool wrap, uint32_t pos, uint32_t tmask, uint32_t tie) {
    std::vector<int64_t> offs;
    for (uint32_t x = 0; x < n; ++x) {
        if (((tmask >> x) & 1u) && x != pos) {
            const int64_t o =
                wrap ? routing_detail::pmod(static_cast<int64_t>(x) - pos, n) : static_cast<int64_t>(x) - pos;
            if (std::find(offs.begin(), offs.end(), o) == offs.end()) {
                offs.push_back(o);
            }
        }
    }
    if (offs.empty()) {
        return 0;
    }
    if (!wrap) {
        int64_t a = 0, b = 0;
        for (auto o : offs) {
            if (o > 0) {
                a = std::max(a, o);
            } else if (o < 0) {
                b = std::max(b, -o);
            }
        }
        return static_cast<uint8_t>(a | (b << 4));
    }
    std::vector<std::pair<int64_t, int64_t>> cands;
    for (int64_t a = 0; a < n; ++a) {
        for (int64_t b = 0; b < static_cast<int64_t>(n) - a; ++b) {
            const bool covers = std::all_of(
                offs.begin(), offs.end(), [&](int64_t o) { return o <= a || static_cast<int64_t>(n) - o <= b; });
            if (covers) {
                cands.emplace_back(a, b);
            }
        }
    }
    int64_t m = INT64_MAX;
    for (auto [a, b] : cands) {
        m = std::min(m, a + b);
    }
    std::erase_if(cands, [&](const auto& p) { return p.first + p.second != m; });
    int64_t mm = INT64_MAX;
    for (auto [a, b] : cands) {
        mm = std::min(mm, std::max(a, b));
    }
    std::erase_if(cands, [&](const auto& p) { return std::max(p.first, p.second) != mm; });
    std::sort(cands.begin(), cands.end());
    const auto [a, b] = cands[cands.size() > 1 ? tie : 0];
    return static_cast<uint8_t>(a | (b << 4));
}

std::vector<uint8_t> span_table(uint32_t n, bool wrap) {
    std::vector<uint8_t> t;
    t.reserve(static_cast<size_t>(n) * (2u << n));
    for (uint32_t pos = 0; pos < n; ++pos) {
        for (uint32_t tm = 0; tm < (1u << n); ++tm) {
            for (uint32_t tie = 0; tie < 2; ++tie) {
                t.push_back(span_entry(n, wrap, pos, tm, tie));
            }
        }
    }
    return t;
}

std::vector<uint8_t> ring_table(uint32_t n, bool wrap) {
    if (!wrap) {
        return {0};
    }
    std::vector<uint8_t> t;
    t.reserve(2u << n);
    for (uint32_t rm = 0; rm < (1u << n); ++rm) {
        for (uint32_t tie = 0; tie < 2; ++tie) {
            t.push_back(span_entry(n, true, 0, rm, tie));
        }
    }
    return t;
}

SpanTables span_subs(uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) {
    return SpanTables{
        .wrap_x = wrap_x, .wrap_y = wrap_y, .xring = ring_table(C, wrap_x), .yring = ring_table(R, wrap_y)};
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> axis_reach(uint32_t n, bool wrap) {
    const auto t = span_table(n, wrap);
    const size_t per = 2u << n;
    std::vector<uint32_t> fwd(n, 0), bwd(n, 0);
    for (uint32_t p = 0; p < n; ++p) {
        for (size_t i = p * per; i < (p + 1) * per; ++i) {
            fwd[p] = std::max<uint32_t>(fwd[p], t[i] & 15u);
            bwd[p] = std::max<uint32_t>(bwd[p], t[i] >> 4);
        }
    }
    return {fwd, bwd};
}

ManifestLayout manifest_layout(uint32_t R, uint32_t C, bool wrap_x, bool wrap_y, uint32_t S) {
    const auto [fx, bx] = axis_reach(C, wrap_x);
    const auto [fy, by] = axis_reach(R, wrap_y);
    // hops from p to q moving + (fwd) or -, nullopt if that way does not lead there
    auto hops = [](int64_t n, bool wrap, int64_t p, int64_t q, bool fwd) -> std::optional<int64_t> {
        const int64_t d = fwd ? q - p : p - q;
        const int64_t k = wrap ? routing_detail::pmod(d, n) : d;
        return k > 0 ? std::optional<int64_t>(k) : std::nullopt;
    };
    auto reaches = [&](int64_t n,
                       bool wrap,
                       const std::vector<uint32_t>& f,
                       const std::vector<uint32_t>& b,
                       int64_t p,
                       int64_t q) {
        const auto kf = hops(n, wrap, p, q, true);
        const auto kb = hops(n, wrap, p, q, false);
        return (kf && *kf <= f[p]) || (kb && *kb <= b[p]);
    };
    struct Axis {
        int64_t n;
        bool w;
        const std::vector<uint32_t>& f;
        const std::vector<uint32_t>& b;
        int64_t no;
        bool wo;
        const std::vector<uint32_t>& fo;
        const std::vector<uint32_t>& bo;
    };
    const std::array<Axis, 2> axes = {
        Axis{C, wrap_x, fx, bx, R, wrap_y, fy, by},  // X
        Axis{R, wrap_y, fy, by, C, wrap_x, fx, bx},  // Y
    };
    int64_t n0 = 0, n1 = 0, n2 = 0;
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            for (size_t ax = 0; ax < 2; ++ax) {
                const auto& A = axes[ax];
                const int64_t pos = ax == 0 ? c : r;
                const int64_t opos = ax == 0 ? r : c;
                for (bool fwd : {true, false}) {
                    const auto& span = fwd ? A.f : A.b;
                    int64_t cont = 0;
                    for (int64_t e = 0; e < A.n; ++e) {
                        const auto k = hops(A.n, A.w, e, pos, fwd);
                        if (k && *k + 1 <= span[e]) {
                            cont += 1;
                            for (int64_t q = 0; q < A.no; ++q) {
                                if (q != opos && reaches(A.no, A.wo, A.fo, A.bo, q, opos)) {
                                    cont += 1;
                                }
                            }
                        }
                    }
                    n0 = std::max(n0, cont);
                }
                for (bool fwd : {true, false}) {
                    const auto& span = fwd ? A.fo : A.bo;
                    int64_t cnt = 0;
                    for (int64_t q = 0; q < A.no; ++q) {
                        const auto k = hops(A.no, A.wo, q, opos, fwd);
                        if (k && *k <= span[q]) {
                            cnt += 1;
                        }
                    }
                    if (fwd) {
                        n1 = std::max(n1, cnt);
                    } else {
                        n2 = std::max(n2, cnt);
                    }
                }
            }
        }
    }
    const int64_t gmax = (static_cast<int64_t>(C) - 1) + (static_cast<int64_t>(R) - 1) - 1;
    const int64_t e0 = n0 * S + gmax + 4, e1 = n1 * S + gmax + 4, e2 = n2 * S + gmax + 4;
    TT_FATAL(e0 + e1 < (1 << 16), "moe_block: manifest ring offsets are 16-bit entry counts ({} + {})", e0, e1);
    return ManifestLayout{
        .ring1_offset = static_cast<uint32_t>(e0),
        .ring2_offset = static_cast<uint32_t>(e0 + e1),
        .total_entries = static_cast<uint32_t>(e0 + e1 + e2)};
}

MeshPlan mesh_plan(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t num_links, bool torus_x, bool torus_y) {
    const auto shape = mesh_device.shape();
    TT_FATAL(shape.dims() == 2, "moe_block: a 2D mesh, got {} dims", shape.dims());
    const auto topo = MeshTopology::of(shape[0], shape[1], torus_x, torus_y);
    auto node = [&](ChipCoord ch) {
        return mesh_device.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(ch.r, ch.c));
    };
    MeshPlan plan{.topo = topo, .num_links = num_links, .chips = {}};
    for (uint32_t r = 0; r < topo.R; ++r) {
        for (uint32_t c = 0; c < topo.C; ++c) {
            const ChipCoord ch{r, c};
            auto& ports = plan.chips[ch];
            for (Dir d : kDirs) {
                const auto peer = topo.neighbor(ch, d);
                if (!peer) {
                    continue;
                }
                TT_FATAL(
                    !tt::tt_fabric::get_neighbor_eth_directions(mesh_device, node(ch), node(*peer)).empty(),
                    "moe_block: ({}, {}) -> ({}, {}) is not a direct link",
                    ch.r,
                    ch.c,
                    peer->r,
                    peer->c);
                auto links = tt::tt_fabric::get_forwarding_link_indices(mesh_device, node(ch), node(*peer));
                TT_FATAL(
                    links.size() >= num_links,
                    "moe_block: ({}, {}) -> ({}, {}) has {} links, {} needed",
                    ch.r,
                    ch.c,
                    peer->r,
                    peer->c,
                    links.size(),
                    num_links);
                links.resize(num_links);
                ports[static_cast<size_t>(d)] = MeshPort{.peer = *peer, .links = std::move(links)};
            }
        }
    }
    return plan;
}

MeshPlan mesh_plan_uniform(const MeshTopology& topo, uint32_t num_links) {
    MeshPlan plan{.topo = topo, .num_links = num_links, .chips = {}};
    std::vector<uint32_t> links(num_links);
    for (uint32_t l = 0; l < num_links; ++l) {
        links[l] = l;
    }
    for (uint32_t r = 0; r < topo.R; ++r) {
        for (uint32_t c = 0; c < topo.C; ++c) {
            const ChipCoord ch{r, c};
            auto& ports = plan.chips[ch];
            for (Dir d : kDirs) {
                if (const auto peer = topo.neighbor(ch, d)) {
                    ports[static_cast<size_t>(d)] = MeshPort{.peer = *peer, .links = links};
                }
            }
        }
    }
    return plan;
}

PortLuts port_luts(const MeshPlan& plan) {
    PortLuts out;
    for (const auto& [ch, ports] : plan.chips) {
        std::array<uint8_t, 8> lut{};
        lut.fill(0xFF);
        uint32_t j = 0;
        for (Dir d : kDirs) {
            if (ports[static_cast<size_t>(d)].has_value()) {
                for (uint32_t li = 0; li < plan.num_links; ++li) {
                    lut[static_cast<size_t>(d) * 2 + li] = static_cast<uint8_t>(j++);
                }
            }
        }
        out.lut[ch] = lut;
        out.num_ports[ch] = j;
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block
