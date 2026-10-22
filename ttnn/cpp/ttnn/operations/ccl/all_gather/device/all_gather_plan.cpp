// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_plan.hpp"

#include "all_gather_device_operation_types.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::ccl::all_gather_plan {

namespace {

// NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES on Wormhole and Blackhole.
constexpr uint32_t kNocWriteAlignment = 16;

// Input pages along [gather dim .. last dim], i.e. what one device contributes to one output row. 0 when a
// row-major row is not a whole number of pages: the walk cannot place them. Specs pad a sharded row to whole
// shards, so this guards against one that does not.
uint32_t pages_per_stripe_row(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end, uint32_t page_bytes) {
    const auto& shape = in.padded_shape();
    const bool tiled = in.layout() == tt::tt_metal::Layout::TILE;
    const auto tile = in.tile();
    uint32_t pages = 1;
    for (int32_t i = dim_from_end; i < 0; i++) {
        if (i == -1) {
            if (tiled) {
                pages *= shape[i] / tile.get_width();
                continue;
            }
            // A row-major row can span several pages (width sharding), so count pages, not elements.
            const uint64_t row_bytes =
                uint64_t{shape[i]} * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(in.data_type()));
            if (row_bytes % page_bytes != 0) {
                return 0;
            }
            pages *= row_bytes / page_bytes;
        } else if (tiled && i == -2) {
            pages *= shape[i] / tile.get_height();
        } else {
            pages *= shape[i];
        }
    }
    return pages;
}

}  // namespace

bool gather_shape_ok(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end, uint32_t N) {
    // Gather-dim padding would need the output's gather-dim extent to be N times the input's padded extent, but it
    // is N times the logical one.
    const uint64_t extent = in.padded_shape()[dim_from_end];
    return in.logical_shape()[dim_from_end] == extent && extent * N <= std::numeric_limits<uint32_t>::max();
}

std::optional<Geometry> make_geometry(const GeoIn& g, bool reads_output) {
    if (g.ci == 0 || g.co == 0 || g.pages_per_stripe_row == 0) {
        return std::nullopt;
    }
    // Every source is read at unit offsets into a unit-aligned slot, so the unit must be a multiple of each source's
    // read alignment (else BH reads (src & ~63) | (dst & 63)) and of the NOC write alignment.
    const uint32_t A = std::lcm(std::lcm(kNocWriteAlignment, g.ri), reads_output ? g.ro : 1u);
    Geometry geo{};
    geo.A = A;
    if (g.ci % A == 0 && g.co % A == 0) {
        // Sub-page: cuts land on any unit boundary, also inside a page.
        geo.Ui = g.ci / A;
        geo.Uo = g.co / A;
    } else if (g.ci == g.co && tt::align(g.ci, A) <= g.ao) {
        // Whole-page: a page moves at pitch align(c, A), and the tail past c lands in the output page's own padding.
        geo.Ui = geo.Uo = tt::align(g.ci, A) / A;
        geo.whole_page = true;
    } else {
        return std::nullopt;
    }

    geo.P = g.P - (g.P % A);
    TT_FATAL(geo.P >= A, "all_gather: fabric payload {} B is smaller than the {} B cut quantum", g.P, A);
    // Pages that are not whole rows (packed size and page grid disagree) or a count past uint32: composite.
    if (g.in_pages % g.pages_per_stripe_row != 0) {
        return std::nullopt;
    }
    const uint64_t S = static_cast<uint64_t>(g.pages_per_stripe_row) * geo.Ui;
    geo.R = g.in_pages / g.pages_per_stripe_row;
    geo.E = (S % geo.Uo == 0) ? geo.Uo : 1;
    const uint64_t elements = geo.R * S / geo.E;
    if (S > std::numeric_limits<uint32_t>::max() || elements > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    geo.S = static_cast<uint32_t>(S);
    geo.elements = static_cast<uint32_t>(elements);
    geo.dst_cross = geo.Uo * A == g.ao;
    geo.src_cross = geo.Ui * A == g.ai;
    return geo;
}

PlanIn make_plan_in(
    const AllGatherParams& args,
    const tt::tt_metal::TensorSpec& in,
    const tt::tt_metal::TensorSpec& out,
    uint32_t avail_cores) {
    return make_plan_in(
        args,
        in,
        out,
        avail_cores,
        tt::tt_metal::hal::get_arch(),
        tt::tt_metal::hal::get_dram_alignment(),
        tt::tt_metal::hal::get_l1_alignment());
}

PlanIn make_plan_in(
    const AllGatherParams& args,
    const tt::tt_metal::TensorSpec& in,
    const tt::tt_metal::TensorSpec& out,
    uint32_t avail_cores,
    tt::ARCH arch,
    uint32_t dram_alignment,
    uint32_t l1_alignment) {
    const uint32_t N = args.num_devices;
    PlanIn p{};
    p.arch = arch;
    p.N = N;
    p.avail_cores = avail_cores;
    p.output_packed_bytes = out.compute_packed_buffer_size_bytes();

    // The walk takes the output page grid to be the input's with only the gather dim scaled by N. Anything else (an
    // output shard that doesn't divide the gathered shape pads it) is a different grid.
    p.shape_ok = gather_shape_ok(in, args.dim_from_end, N);
    if (p.shape_ok) {
        auto gathered = in.padded_shape();
        gathered[args.dim_from_end] *= N;
        p.shape_ok = out.padded_shape() == gathered;
    }
    if (p.shape_ok) {
        const auto ci = static_cast<uint32_t>(in.compute_page_size_bytes());
        const auto co = static_cast<uint32_t>(out.compute_page_size_bytes());
        const uint32_t pps = ci ? pages_per_stripe_row(in, args.dim_from_end, ci) : 0;
        p.shape_ok = pps != 0;
        const uint32_t ri = in.memory_config().is_dram() ? dram_alignment : l1_alignment;
        const uint32_t ro = out.memory_config().is_dram() ? dram_alignment : l1_alignment;
        p.geo = GeoIn{
            .ci = ci,
            .co = co,
            .ai = tt::align(ci, ri),
            .ao = tt::align(co, ro),
            .ri = ri,
            .ro = ro,
            .in_pages = ci ? static_cast<uint32_t>(in.compute_packed_buffer_size_bytes() / ci) : 0,
            .pages_per_stripe_row = pps,
            .N = N,
            .P = static_cast<uint32_t>(args.packet_size)};
    }

    p.true_2d = args.is_true_2d();
    if (p.true_2d) {
        // One multicast core per link serves both axes, so the axis with fewer links sets the count.
        p.links = std::min(args.axis_num_links[0], args.axis_num_links[1]);
    } else {
        const uint32_t axis = args.get_1d_axis();
        p.ring = tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis]);
        p.links = args.axis_num_links[axis];
        // Multicast cannot route along an axis a 2D view bends, and neighbor exchange only forwards one hop.
        p.forced_unicast = (tt::tt_fabric::is_2d_fabric_config(args.fabric_config) && !args.axis_is_straight[axis]) ||
                           args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
    }
    return p;
}

Plan make_plan(const PlanIn& in) {
    Plan plan{};
    plan.algo = AgAlgo::Composite;
    plan.N = in.N;
    // Two devices are one hop apart both ways, so a 2-device ring would send both directions to the same neighbor.
    plan.ring = in.ring && in.N > 2;
    if (!in.shape_ok) {
        plan.composite_reason = "the output pages are not the input page grid scaled on the gather dim";
        return plan;
    }

    // Unicast needs a 1D axis; multicast cannot run on a forced-unicast fabric.
    const auto geo_u = in.true_2d ? std::nullopt : make_geometry(in.geo, /*reads_output=*/true);
    const auto geo_m = in.forced_unicast ? std::nullopt : make_geometry(in.geo, /*reads_output=*/false);
    // Links each algorithm can use: unicast takes a core per direction, and no slice may be empty. 0 = cannot run.
    auto effective_links = [&](const std::optional<Geometry>& geo, uint32_t cores_per_link) -> uint32_t {
        return geo ? std::min({in.links, in.avail_cores / cores_per_link, geo->elements}) : 0;
    };
    const uint32_t links_u = effective_links(geo_u, 2);
    const uint32_t links_m = effective_links(geo_m, 1);
    if (links_u == 0 && links_m == 0) {
        auto why = [](const std::optional<Geometry>& geo, bool barred, const char* bar) {
            if (barred) {
                return bar;
            }
            if (!geo) {
                return "cannot cut these pages into aligned units";
            }
            return geo->elements == 0 ? "has nothing to move" : "has no free worker cores";
        };
        plan.composite_reason = fmt::format(
            "unicast {}; multicast {}",
            why(geo_u, in.true_2d, "needs a 1D axis"),
            why(geo_m, in.forced_unicast, "cannot route on this fabric axis"));
        return plan;
    }
    const bool unicast = (links_u > 0 && links_m > 0) ? prefer_unicast(
                                                            in.arch,
                                                            plan.ring,
                                                            in.N,
                                                            links_u,
                                                            geo_u->P,
                                                            in.output_packed_bytes / links_u,
                                                            uint64_t{in.geo.ci} * in.geo.pages_per_stripe_row)
                                                      : links_u > 0;

    plan.algo = unicast ? AgAlgo::Unicast : AgAlgo::Multicast;
    plan.geo = unicast ? *geo_u : *geo_m;
    plan.links = unicast ? links_u : links_m;
    plan.slices = make_slices(plan.geo.elements, plan.links);
    return plan;
}

bool prefer_unicast(
    tt::ARCH arch,
    bool ring,
    uint32_t N,
    uint32_t links,
    uint32_t P,
    uint64_t busiest_link_bytes,
    uint64_t stripe_row_bytes) {
    // Row-major gathers of small row segments (a few hundred bytes) are bound by the packet rate either way (at most
    // 4 segments per packet); composite might win there (unmeasured).
    switch (arch) {
        case tt::ARCH::BLACKHOLE: {
            // Measured on 8x p150b at 2 links, payload 15168, 8 KB - 13 MB per link. 1 link is unmeasured; past 2,
            // unicast's DRAM reads are predicted to bind.
            // small: a stripe is under 6 full packets per link, so latency decides (ring unicast 5-18% faster, line
            // multicast up to 34%: unicast relays N - 1 hops store-and-forward).
            // wide: unicast pays per run, and a DRAM run is at most one bank's share of a stripe row. Runs >= 8 KB
            // favor unicast (5-31%), 4 KB only on the ring, <= 2 KB multicast (up to 2x).
            if (links > 2) {
                return false;
            }
            const bool small = busiest_link_bytes < uint64_t{N} * 6 * P;
            constexpr uint64_t kDramBanks = 8;
            const bool wide = stripe_row_bytes / kDramBanks >= (ring ? 4 : 8) * 1024;
            return ring ? small || wide : !small && wide;
        }
        case tt::ARCH::WORMHOLE_B0:
            // Unverified with these kernels: fit to the earlier (mux-based) ones on T3000. Ring: the fabric relays
            // for free. Two devices: unicast never relays. 3-6: multicast's edge shrinks with N. 7+: near tie, so
            // size decides.
            if (ring) {
                return false;
            }
            if (N <= 2) {
                return true;
            }
            if (N <= 6) {
                return false;
            }
            return busiest_link_bytes >= 3'000'000;
        default: return false;  // uncalibrated arch
    }
}

std::vector<Slice> make_slices(uint32_t elements, uint32_t links) {
    TT_FATAL(links > 0 && links <= elements, "all_gather: cannot cut {} elements into {} slices", elements, links);
    const uint32_t base = elements / links;
    const uint32_t rem = elements % links;
    std::vector<Slice> slices(links);
    for (uint32_t l = 0; l < links; ++l) {
        slices[l] = Slice{.e0 = (l * base) + std::min(l, rem), .ne = base + (l < rem ? 1 : 0)};
    }
    return slices;
}

UnicastDir unicast_schedule(const Plan& plan, uint32_t dev, uint32_t link, uint32_t dir) {
    const uint32_t N = plan.N;
    TT_FATAL(dev < N && link < plan.slices.size() && dir < 2, "all_gather: bad unicast core {} {} {}", dev, link, dir);
    const bool fwd = dir == 0;
    const uint32_t ne = plan.slices[link].ne;
    const uint32_t E = plan.geo.E;
    // Stripes a direction sends: ring N/2; line forward d+1, backward N-d; none from a line endpoint's dead side.
    auto iters = [&](bool forward) -> uint32_t {
        if (plan.ring) {
            return N / 2;
        }
        return forward ? (dev + 1 < N ? dev + 1 : 0) : (dev > 0 ? N - dev : 0);
    };

    UnicastDir u{};
    u.I = iters(fwd);
    // Iteration i sends stripe dev - i (forward) or dev + i (backward): what upstream sent in iteration i - 1.
    u.stripe_step = fwd ? N - 1 : 1;
    u.iter_units = ne * E;
    // Even ring: the antipode is reached both ways, so each direction takes half of that stripe.
    const bool even_split = plan.ring && N % 2 == 0;
    u.last_skip = (even_split && !fwd) ? ne / 2 : 0;
    u.last_take = u.I == 0 ? 0 : (even_split ? (fwd ? ne / 2 : ne - (ne / 2)) : ne);
    u.last_units = u.last_take * E;
    // Units the same-direction upstream core sends here; it has the same slice, so the same last_units.
    if (plan.ring) {
        u.recv_units = even_split ? ((N / 2 - 1) * u.iter_units) + u.last_units : ((N - 1) / 2) * u.iter_units;
    } else {
        u.recv_units = (fwd ? dev : N - 1 - dev) * u.iter_units;
    }
    // Iteration 0 carries this device's own stripe; the forward core copies it, unless it has nothing to send.
    u.local_copy = fwd ? iters(true) > 0 : iters(true) == 0;
    return u;
}

McastRoutes mcast_routes(
    const Plan& plan,
    std::array<uint32_t, 2> mesh_shape,
    std::array<uint32_t, 2> my_coord,
    std::optional<uint32_t> axis,
    std::array<bool, 2> axis_ring,
    std::array<uint32_t, 4> phys_dir) {
    // Hops per logical direction for route variant 0 and 1. Axis 1 runs E (forward) / W, axis 0 runs S (forward) / N.
    std::array<uint32_t, 4> h0{}, h1{};
    bool alternates = false;
    uint32_t devices = 1;
    for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t size = mesh_shape[a];
        if (size <= 1 || axis.value_or(a) != a) {
            continue;
        }
        TT_FATAL(size < 256 && my_coord[a] < size, "all_gather: bad mesh axis {}: {} of {}", a, my_coord[a], size);
        devices *= size;
        const bool ring = axis_ring[a];
        const uint32_t fwd = ring ? size / 2 : size - 1 - my_coord[a];
        const uint32_t bwd = ring ? size - 1 - fwd : my_coord[a];
        // Even ring: forward reaches one device further than backward; alternating the split by packet balances them.
        const bool alt = ring && size % 2 == 0;
        alternates = alternates || alt;
        const uint32_t f = a == 1 ? kEast : kSouth;
        const uint32_t b = a == 1 ? kWest : kNorth;
        h0[f] = fwd;
        h0[b] = bwd;
        h1[f] = alt ? bwd : fwd;
        h1[b] = alt ? fwd : bwd;
    }
    TT_FATAL(devices == plan.N, "all_gather: mesh axes hold {} devices, the plan {}", devices, plan.N);

    // E/W connections are lines; N/S connections are spines that branch E and W at every hop, so each variant
    // covers every other device exactly once.
    auto range = [&](const std::array<uint32_t, 4>& h, uint32_t d) {
        uint32_t r = 0;
        auto put = [&](uint32_t logical) {
            if (h[logical] > 0) {  // an absent branch must not clobber a live slot
                r |= h[logical] << (8 * phys_dir[logical]);
            }
        };
        put(d);
        if (d == kNorth || d == kSouth) {
            put(kEast);
            put(kWest);
        }
        return r;
    };
    McastRoutes routes{.conns = {}, .alternates = alternates};
    for (uint32_t d = kEast; d <= kSouth; ++d) {
        if (h0[d] > 0) {
            routes.conns.push_back(McastConn{.dir = d, .range_v0 = range(h0, d), .range_v1 = range(h1, d)});
        }
    }
    return routes;
}

}  // namespace ttnn::operations::ccl::all_gather_plan
