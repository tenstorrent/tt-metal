// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_program.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <numeric>
#include <set>
#include <utility>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/kernel_lib/mcast/host/mcast.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace head_program_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::distributed::MeshDevice;

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/";

// se_engine3's constants
constexpr uint32_t kNws = 3;
constexpr uint32_t kXready = 0, kWv = 1, kWfree = 1 + kNws;
constexpr uint32_t kXld = 1 + 2 * kNws;
constexpr uint32_t kNsemOwn = 2 + 2 * kNws;
constexpr uint32_t kRxr = 0, kRwr = 16, kRxs = 32, kXrl = 96, kXack = 128, kWack = 144;
enum SeCb : uint8_t { XIN = 0, WIN = 1, HOWN = 2, HALL = 3, OUT = 4, ACC = 5 };
constexpr uint8_t kRsHold = 6, kRsFinb = 7, kRsArrf = 8, kRsArrb = 9, kRsFinc = 10, kRsOutf = 11, kRsPsync = 12,
                  kRsFfree = 20;
constexpr uint8_t kLatCb = 13, kPrepPu = 14, kPrepPo = 15, kPrepLrdy = 16, kXfreeCb = 21;
constexpr uint8_t kRtrRx = 17, kRtrRw = 18, kRtrRo = 19;
// x_gather3's
constexpr uint32_t kIncEvery = 8;
constexpr uint8_t kXgCb = 0;
// router_stream's CB ids
enum RtCb : uint8_t { RT_HOLD = 2, ARRF = 3, ARRB = 4, BIAS = 5, SC = 6, PSYNC = 8, BEST = 9, IDX = 11, RDY = 12 };
constexpr uint32_t kPlanWords = 1024;

uint32_t u32(size_t v) { return static_cast<uint32_t>(v); }
using head_math::al16;
using head_math::al64;
int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges);
}

KernelDescriptor kernel(const std::string& file, const CoreRangeSet& cores, KernelDescriptor::ConfigDescriptor cfg) {
    KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.config = std::move(cfg);
    return k;
}

DataMovementConfigDescriptor dm(DataMovementProcessor risc, NOC noc) {
    return DataMovementConfigDescriptor{.processor = risc, .noc = noc};
}

void define(KernelDescriptor& k, const std::string& name, const std::string& value = "") {
    k.defines.emplace_back(name, value);
}
std::string uv(int64_t v) { return std::to_string(v) + "u"; }
std::string iv(int64_t v) { return std::to_string(v); }

void append_accessor(std::vector<uint32_t>& ct, const Buffer* b) { tt::tt_metal::TensorAccessorArgs(*b).append_to(ct); }
size_t accessor_len(const Buffer* b) {
    std::vector<uint32_t> v;
    append_accessor(v, b);
    return v.size();
}

// A float as Python's repr then "f" (se_engine3._f): the shortest round-trip decimal, fixed notation, ".0" if whole.
std::string float_literal(float v) {
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed);
    std::string s(buf, res.ptr);
    if (s.find('.') == std::string::npos) {
        s += ".0";
    }
    return s + "f";
}

enum class XDir : uint8_t { Fwd = 0, Bwd = 1 };
inline XDir opp(XDir d) { return d == XDir::Fwd ? XDir::Bwd : XDir::Fwd; }
inline RsDir rs_dir(XDir d) { return d == XDir::Fwd ? RsDir::Fwd : RsDir::Bwd; }

using Coord = std::pair<uint32_t, uint32_t>;  // (mesh row, column)

int64_t pmod(int64_t a, int64_t b) { return ((a % b) + b) % b; }

// The gather's schedule: (shards toward p + 1, toward p - 1), own first: (position, part)
std::pair<std::vector<Coord>, std::vector<Coord>> gather_schedule(uint32_t p, uint32_t G, bool ring, bool balance) {
    std::vector<Coord> fwd, bwd;
    if (ring && balance && G % 2 == 0 && G >= 4) {
        const uint32_t h = G / 2;
        for (uint32_t i = 0; i < h; ++i) {
            fwd.emplace_back(pmod(int64_t(p) - i, G), i == h - 1 ? 1 : 0);
            bwd.emplace_back(pmod(int64_t(p) + i, G), i == h - 1 ? 2 : 0);
        }
        return {fwd, bwd};
    }
    if (ring) {
        const uint32_t kf = G / 2, kb = G - 1 - kf;
        for (uint32_t i = 0; i < kf; ++i) {
            fwd.emplace_back(pmod(int64_t(p) - i, G), 0);
        }
        for (uint32_t i = 0; i < kb; ++i) {
            bwd.emplace_back(pmod(int64_t(p) + i, G), 0);
        }
        return {fwd, bwd};
    }
    if (p < G - 1) {
        for (uint32_t i = 0; i <= p; ++i) {
            fwd.emplace_back(p - i, 0);
        }
    }
    if (p > 0) {
        for (uint32_t i = 0; i < G - p; ++i) {
            bwd.emplace_back(p + i, 0);
        }
    }
    return {fwd, bwd};
}

}  // namespace head_program_detail

using namespace head_program_detail;

// ---------------------------------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------------------------------

// A gather ring a chip.
struct GatherRing {
    uint32_t p = 0;
    std::optional<Coord> next, prev;
    std::array<std::vector<Coord>, 2> sched;     // [fwd, bwd]: (position, part)
    std::array<std::vector<uint32_t>, 2> links;  // fabric link indices a direction (empty: no port traffic)
    const std::optional<Coord>& peer(XDir d) const { return d == XDir::Fwd ? next : prev; }
};

struct GatherChip {
    GatherRing rg;
    std::vector<std::tuple<XDir, uint32_t, CoreCoord>> ports;  // (direction, link, logical core), plan order
    std::vector<CoreCoord> copy;
    CoreCoord port(XDir d, uint32_t l) const {
        for (const auto& [dd, ll, c] : ports) {
            if (dd == d && ll == l) {
                return c;
            }
        }
        TT_THROW("fused_head: no port ({}, {})", static_cast<int>(d), l);
    }
};

// A routing exchange sender key: ("x", direction, link) or ("y", 2 | 3), ordered as the reference's tuples.
struct SendKey {
    bool y = false;
    uint32_t a = 0;  // x: 0 bwd / 1 fwd (sort order); y: 2 | 3
    uint32_t l = 0;
    auto operator<=>(const SendKey&) const = default;
    XDir dir() const { return a == 1 ? XDir::Fwd : XDir::Bwd; }
    static SendKey x(XDir d, uint32_t l) { return {false, d == XDir::Fwd ? 1u : 0u, l}; }
};

class HeadModel {
public:
    HeadProgramConfig cfg;
    MeshDevice* mesh = nullptr;
    uint32_t R = 0, C = 0, N = 0, G = 0;
    CoreCoord grid;
    uint32_t MX = 0;  // the mover column
    bool ring = false, wrap_y = false;
    uint32_t L = 0;  // links (the gather's)
    uint32_t payload = 0;
    std::optional<RsPlan> rs;
    std::optional<SeEngineSizing> eng;
    // gather
    uint32_t klt = 0, sps = 0, rp = 0, group = 4;
    std::map<Coord, GatherChip> chips;
    CoreCoord watcher;
    std::map<std::pair<uint32_t, uint32_t>, XDir> ties;  // (p, b) of a ring's far side -> the fabric's way
    // router
    std::vector<CoreCoord> rcores;
    uint32_t RB = 0, KT = 0, ET = 0, K = 0, NS = 1;
    Regions roff;
    int64_t rbytes = 0, pbuf_sub = 0;
    // the mailbox words
    uint32_t HS = 16, fslots = 16, lw_off = 0, port_words = 16, zw_off = 0, mbox_words = 0;
    // routing
    uint32_t S = 0, EPC = 0, A = 0, Lr = 2, M = 0, NF = 16, rows_cap = 0, inst_cap = 0;
    int64_t pl_step = 0, plan_l1 = 0, HBP = 0, BB = 0, BBP = 0, PCAP = 0, MAXP = 0;
    std::vector<int64_t> pbases_off;  // the planner instances' regions, arena offsets
    int64_t xbase_off = 0, fbase_off = 0;
    int64_t xg_off = 0, xg_bytes = 0;  // the x-gather packet staging on the port cores (kXgCb), arena offset / bytes
    uint32_t nx = 0, ny = 0;
    PortLuts luts;
    std::vector<SendKey> fkeys;
    std::map<std::pair<uint32_t, SendKey>, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>> fwd;  // forwards
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> fwd_at;
    std::map<uint32_t, std::map<SendKey, std::vector<uint32_t>>> serves;  // per chip index
    int64_t arena_bytes = 0;  // the arena's bytes a core (the engine's, grown to the router cores')

    CoreCoord virt(const CoreCoord& c) const { return mesh->worker_core_from_logical_core(c); }
    CoreCoord evirt(uint32_t cx, uint32_t cy) const { return virt(CoreCoord(cx, cy)); }
    tt::tt_fabric::FabricNodeId node(const Coord& c) const {
        return mesh->get_fabric_node_id(ttnn::MeshCoordinate(c.first, c.second));
    }
    tt::tt_fabric::FabricNodeId node_idx(uint32_t i) const { return node({i / C, i % C}); }
    const GatherChip& chip(const Coord& c) const { return chips.at(c); }
    // head_words' order
    static uint32_t word_index(const std::string& n) {
        static const std::vector<std::string> names = {
            "mbox", "arr", "xg", "rt", "slot_words", "sfree", "ra_c", "ra_p", "ra_s", "yhello", "ewords"};
        const auto it = std::find(names.begin(), names.end(), n);
        TT_FATAL(it != names.end(), "fused_head: no persistent words '{}'", n);
        return u32(it - names.begin());
    }
    Coord coord_of_pos(const Coord& c, uint32_t pos) const {
        for (const auto& [k, ch] : chips) {
            if (k.first == c.first && ch.rg.p == pos) {
                return k;
            }
        }
        TT_THROW("fused_head: no chip at position {}", pos);
    }
    uint32_t pos(const Coord& c) const { return chips.at(c).rg.p; }
    // the Y hello (ra_y_send): a chip with a remote Y destination sends one to its first Y direction's neighbour (the
    // row the BRISC's fabric connection reaches) and waits for the ones that land on it. The row it sends to, if any:
    std::optional<uint32_t> y_hello_to(const Coord& c) const {
        const uint32_t me = c.first * C + c.second;
        for (const auto& [k, ds] : serves.at(me)) {
            if (k.y) {
                if (ds.size() == (ds.back() == me ? 1u : 0u)) {
                    return std::nullopt;  // (no remote destination: no connection, no hello)
                }
                return u32(pmod(int64_t(c.first) + (k.a == 2 ? 1 : -1), R));
            }
        }
        return std::nullopt;
    }
    // the hellos landing on chip c a call: the rows of its column whose hello goes to its row. A ring (and R 2): one
    // each. A line (R > 2, no wrap): every row's first Y direction is Y+ except the far end's (Y- only) - the near
    // end gets none, the row before the far end two, the others one
    uint32_t y_hellos_in(const Coord& c) const {
        uint32_t n = 0;
        for (uint32_t r = 0; r < R; ++r) {
            n += r != c.first && y_hello_to({r, c.second}) == c.first;
        }
        return n;
    }
    CoreRangeSet engine_cores() const {
        return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(eng->GX - 1, eng->GY - 1)));
    }
    std::vector<CoreCoord> all_ports() const {
        std::set<std::pair<size_t, size_t>> s;
        for (const auto& [_, ch] : chips) {
            for (const auto& [d, l, c] : ch.ports) {
                s.insert({c.x, c.y});
            }
        }
        std::vector<CoreCoord> out;
        for (const auto& [x, y] : s) {
            out.emplace_back(x, y);
        }
        return out;
    }

    // ---- x_gather3 ----
    XDir ring_dir(uint32_t p, uint32_t b) const {
        const uint32_t df = pmod(int64_t(b) - p, G);
        if (df != G - df) {
            return df < G - df ? XDir::Fwd : XDir::Bwd;
        }
        return ties.at({p, b});
    }
    struct PortPlan {
        std::optional<Coord> peer, up;
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> mine;  // (shard, part, upstream base, packets)
        uint32_t up_ipc = 0, expect_in = 0;
        bool has_up = false;
    };
    uint32_t chunk_packets(uint32_t v, uint32_t part) const;
    PortPlan port_plan(const Coord& coord, XDir d, uint32_t v) const;
    std::pair<std::vector<uint32_t>, std::map<uint32_t, std::vector<std::tuple<XDir, uint32_t, uint32_t, uint32_t>>>>
    shard_order(const Coord& coord) const;
    std::vector<uint32_t> ranks(const Coord& coord) const;
    std::vector<uint32_t> tables(const Coord& coord) const;
    // ---- router_stream ----
    std::vector<uint32_t> port_layout(uint32_t l) const;
    // HeadSE3.logits_priority: the ports send the logits ahead of the x gather
    bool logits_first() const {
        if (R * C <= 8) {
            return false;
        }
        if (cfg.shape.latent && ring) {
            return false;
        }
        return true;
    }
    XDir rdir(uint32_t p, uint32_t b) const { return ring ? ring_dir(p, b) : (b > p ? XDir::Fwd : XDir::Bwd); }
    uint32_t hops(uint32_t p, uint32_t b) const {
        return ring ? std::min(pmod(int64_t(b) - p, G), pmod(int64_t(p) - b, G)) : (b > p ? b - p : p - b);
    }
    std::vector<uint32_t> dests(uint32_t p, XDir d) const;
    std::vector<uint32_t> arrival_order(uint32_t p) const;
    // ---- routing_stream ----
    uint32_t first_hop(uint32_t s, uint32_t d) const;
    std::map<uint32_t, XDir> xdir(const Coord& c) const;
    std::vector<std::pair<uint32_t, SendKey>> path(uint32_t s, uint32_t d) const;
    std::map<SendKey, std::vector<uint32_t>> serve(const Coord& c) const;
    CoreCoord core_of(const Coord& c, const SendKey& k) const { return k.y ? watcher : chips.at(c).port(k.dir(), k.l); }
    int64_t buf_off(const Coord& c, const SendKey& k) const;
};

uint32_t HeadModel::chunk_packets(uint32_t v, uint32_t part) const {
    uint32_t rows = 0;
    for (uint32_t lr = 0; lr < eng->CR; ++lr) {
        if ((lr / 2) % L == v && (part == 0 || (lr & 1) == part - 1)) {
            ++rows;
        }
    }
    return rows * sps * cdiv(eng->KB, rp);
}

HeadModel::PortPlan HeadModel::port_plan(const Coord& coord, XDir d, uint32_t v) const {
    auto incs = [](uint32_t n) { return n / kIncEvery + (n % kIncEvery ? 1 : 0); };
    const auto& rg = chips.at(coord).rg;
    PortPlan pl;
    pl.peer = rg.peer(d);
    pl.up = d == XDir::Fwd ? rg.prev : rg.next;
    std::vector<Coord> mine = pl.peer ? rg.sched[static_cast<int>(d)] : std::vector<Coord>{};
    std::vector<Coord> ups = pl.up ? chips.at(*pl.up).rg.sched[static_cast<int>(d)] : std::vector<Coord>{};
    std::vector<uint32_t> up_base;
    uint32_t acc = 0;
    for (const auto& [_, part] : ups) {
        up_base.push_back(acc);
        acc += incs(chunk_packets(v, part));
    }
    for (size_t k = 0; k < mine.size(); ++k) {
        const auto [q, part] = mine[k];
        pl.mine.emplace_back(q, part, k > 0 ? up_base[k - 1] : 0, chunk_packets(v, part));
    }
    pl.up_ipc = acc;
    pl.expect_in = ups.empty() ? 0 : eng->NCH * acc;
    pl.has_up = !ups.empty();
    return pl;
}

std::pair<std::vector<uint32_t>, std::map<uint32_t, std::vector<std::tuple<XDir, uint32_t, uint32_t, uint32_t>>>>
HeadModel::shard_order(const Coord& coord) const {
    const auto& rg = chips.at(coord).rg;
    const uint32_t own = (!rg.sched[0].empty() ? rg.sched[0] : rg.sched[1])[0].first;
    std::map<uint32_t, std::vector<std::tuple<XDir, uint32_t, uint32_t, uint32_t>>> carry;
    std::vector<uint32_t> order_seen;
    for (XDir d : {XDir::Fwd, XDir::Bwd}) {
        const auto& up = d == XDir::Fwd ? rg.prev : rg.next;
        if (!up) {
            continue;
        }
        const auto& ups = chips.at(*up).rg.sched[static_cast<int>(d)];
        for (uint32_t j = 0; j < ups.size(); ++j) {
            const auto [q, part] = ups[j];
            for (uint32_t v = 0; v < L; ++v) {
                if (chunk_packets(v, part)) {
                    carry[q].emplace_back(d, v, j, u32(ups.size()));
                }
            }
        }
    }
    std::vector<uint32_t> rest;
    for (const auto& [q, _] : carry) {
        rest.push_back(q);
    }
    auto key = [&](uint32_t q) {
        uint32_t mj = 0;
        for (const auto& [d, v, j, n] : carry.at(q)) {
            mj = std::max(mj, j);
        }
        return std::make_pair(mj, q);
    };
    std::stable_sort(rest.begin(), rest.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
    std::vector<uint32_t> out = {own};
    out.insert(out.end(), rest.begin(), rest.end());
    return {out, carry};
}

std::vector<uint32_t> HeadModel::ranks(const Coord& coord) const {
    const auto shards = shard_order(coord).first;
    std::vector<uint32_t> r(G, 0);
    for (uint32_t k = 0; k < shards.size(); ++k) {
        r[shards[k]] = k;
    }
    return r;
}

std::vector<uint32_t> HeadModel::tables(const Coord& coord) const {
    const auto& e = *eng;
    const uint32_t nf = e.NSG / 2;
    std::vector<uint32_t> out = ranks(coord);
    out.insert(out.end(), e.fam0.begin(), e.fam0.end());
    out.insert(out.end(), e.fam1.begin(), e.fam1.end());
    for (uint32_t c = 0; c < e.GX; ++c) {
        const uint32_t n0 = e.in_fam0(c) ? nf / e.S0 : 0, n1 = e.in_fam1(c) ? nf / e.S1 : 0;
        out.push_back(n0 + n1);
        out.push_back(n0);
    }
    for (uint32_t c = 0; c < e.GX; ++c) {
        out.push_back(u32(evirt(c, 0).x));
    }
    for (uint32_t cy = 0; cy < e.GY; ++cy) {
        out.push_back(u32(evirt(0, cy).y));
    }
    return out;
}

std::vector<uint32_t> HeadModel::port_layout(uint32_t l) const {
    std::vector<uint32_t> out;
    for (uint32_t j = 0; j < rcores.size(); ++j) {
        if (j % L == l) {
            out.push_back(j);
        }
    }
    return out;
}

std::vector<uint32_t> HeadModel::dests(uint32_t p, XDir d) const {
    std::vector<uint32_t> bs;
    for (uint32_t b = 0; b < G; ++b) {
        if (b != p && rdir(p, b) == d) {
            bs.push_back(b);
        }
    }
    std::stable_sort(bs.begin(), bs.end(), [&](uint32_t a, uint32_t b) { return hops(p, a) > hops(p, b); });
    return bs;
}

std::vector<uint32_t> HeadModel::arrival_order(uint32_t p) const {
    std::vector<uint32_t> srcs;
    for (uint32_t q = 0; q < G; ++q) {
        if (q != p) {
            srcs.push_back(q);
        }
    }
    auto key = [&](uint32_t q) {
        const auto ds = dests(q, rdir(q, p));
        const uint32_t idx = u32(std::find(ds.begin(), ds.end(), p) - ds.begin());
        return std::make_tuple(idx, hops(q, p), q);
    };
    std::stable_sort(srcs.begin(), srcs.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
    std::vector<uint32_t> out;
    for (uint32_t q : srcs) {
        out.push_back(q < p ? q : q - 1);
    }
    return out;
}

uint32_t HeadModel::first_hop(uint32_t s, uint32_t d) const {
    const int64_t sr = s / C, sc = s % C, dr = d / C, dc = d % C;
    if (sr != dr) {
        if (wrap_y) {
            return 2 * pmod(dr - sr, R) < int64_t(R) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if (ring) {
        return 2 * pmod(dc - sc, C) <= int64_t(C) ? 0 : 1;
    }
    return dc > sc ? 0 : 1;
}

std::map<uint32_t, XDir> HeadModel::xdir(const Coord& c) const {
    std::map<uint32_t, XDir> out;
    const auto& rg = chips.at(c).rg;
    for (XDir d : {XDir::Fwd, XDir::Bwd}) {
        const auto& peer = rg.peer(d);
        if (peer && !rg.links[static_cast<int>(d)].empty()) {
            const bool fwd = ring ? pmod(int64_t(peer->second) - c.second, C) == 1 : peer->second > c.second;
            out[fwd ? 0 : 1] = d;
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, SendKey>> HeadModel::path(uint32_t s, uint32_t d) const {
    std::vector<std::pair<uint32_t, SendKey>> steps;
    const uint32_t sr = s / C, sc = s % C, dr = d / C, dc = d % C;
    Coord cur{sr, sc};
    if (sr != dr) {
        steps.emplace_back(s, SendKey{true, first_hop(s, d), 0});
        cur = {dr, sc};
    }
    while (cur.second != dc) {
        const uint32_t ci = cur.first * C + cur.second;
        const uint32_t h = first_hop(ci, d);
        const auto xd = xdir(cur);
        TT_FATAL(xd.count(h), "fused_head: chip ({}, {}): no X port toward {}", cur.first, cur.second, d);
        const uint32_t l = (d % Lr) < L ? d % Lr : 0;
        steps.emplace_back(ci, SendKey::x(xd.at(h), l));
        cur = {cur.first, u32(pmod(int64_t(cur.second) + (h == 0 ? 1 : -1), C))};
    }
    return steps;
}

std::map<SendKey, std::vector<uint32_t>> HeadModel::serve(const Coord& c) const {
    const uint32_t me = c.first * C + c.second;
    std::map<SendKey, std::vector<uint32_t>> out;
    for (uint32_t dd = 0; dd < N; ++dd) {
        if (dd != me) {
            out[path(me, dd)[0].second].push_back(dd);
        }
    }
    std::vector<SendKey> ykeys;
    for (const auto& [k, _] : out) {
        if (k.y) {
            ykeys.push_back(k);
        }
    }
    out[ykeys.empty() ? out.begin()->first : ykeys[0]].push_back(me);
    TT_FATAL(ykeys.size() <= 2, "fused_head: chip ({}, {}): {} Y directions", c.first, c.second, ykeys.size());
    return out;
}

int64_t HeadModel::buf_off(const Coord& c, const SendKey& k) const {
    if (!k.y) {
        return xbase_off;
    }
    const auto& sv = serves.at(c.first * C + c.second);
    uint32_t i = 0;
    for (const auto& [kk, _] : sv) {
        if (kk.y) {
            if (kk == k) {
                return int64_t(i) * ny * PCAP;
            }
            ++i;
        }
    }
    TT_THROW("fused_head: no Y sender");
}

std::shared_ptr<const HeadModel> make_head_model(
    MeshDevice& mesh, const HeadProgramConfig& config, const PortLuts& luts) {
    auto m = std::make_shared<HeadModel>();
    HeadModel& h = *m;
    h.cfg = config;
    h.mesh = &mesh;
    const HeadShape& sh = config.shape;
    const HeadLayout& lay = config.lay;
    h.R = mesh.shape()[0];
    h.C = mesh.shape()[1];
    h.N = h.R * h.C;
    h.G = sh.G;
    TT_FATAL(
        h.C == sh.G && h.R == sh.R,
        "fused_head: the plan's mesh {}x{} is not the device's {}x{}",
        sh.R,
        sh.G,
        h.R,
        h.C);
    h.grid = mesh.compute_with_storage_grid_size();
    h.MX = std::min<uint32_t>(11, h.grid.x - 1);
    TT_FATAL(h.grid.x - h.MX == 1, "fused_head: the movers are one column (grid {}x{})", h.grid.x, h.grid.y);
    h.ring = sh.ring && sh.G > 2;
    h.wrap_y = sh.sp_ring && sh.R > 2;
    h.L = sh.links;
    h.payload = u32(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    h.luts = luts;
    const uint32_t G = h.G, GX = lay.GX, GY = lay.GY, MT = lay.MT;

    // the engine (HeadSE3.__init__'s build at the planned layout)
    h.rs.emplace(G, h.ring, GX, GY, MT, sh.H / 32, sh.rs_dim(), sh.links, kDestTiles / MT, lay.compact);
    SeEngineArgs ea;
    ea.rows = sh.rows;
    ea.H = sh.H;
    ea.I = sh.I;
    ea.GX = GX;
    ea.GY = GY;
    ea.MT = MT;
    ea.KB = lay.KB;
    ea.shards = G;
    ea.rs = &*h.rs;
    if (sh.latent) {
        ea.latent = *sh.latent / 32 / G;
    }
    ea.router = std::make_pair(sh.E / 32, sh.H / G / 32);  // (the router logits on the matmul cores)
    ea.arena_max = arena_cap_of(config.l1_free, config.cb_floor);
    ea.w_dtype = sh.w_dtype;
    ea.xring = lay.xdram ? 2 : 0;
    h.eng.emplace(ea);
    const SeEngineSizing& e = *h.eng;

    // the movers: ports, the watcher, the router cores (HeadSE3)
    const uint32_t nrt = sh.rows / 32 / G;
    if (nrt + 2 * h.L + 1 <= h.grid.y) {
        for (uint32_t cy = h.grid.y - nrt; cy < h.grid.y; ++cy) {
            h.rcores.emplace_back(h.MX, cy);
        }
        h.watcher = CoreCoord(h.MX, h.grid.y - nrt - 1);
    } else {
        std::vector<CoreCoord> free;
        for (uint32_t y = GY; y < h.grid.y; ++y) {
            for (uint32_t c = 0; c < GX; ++c) {
                free.emplace_back(c, y);
            }
        }
        TT_FATAL(nrt <= free.size(), "fused_head: no cells for {} router cores", nrt);
        h.rcores.assign(free.begin(), free.begin() + nrt);
        h.watcher = CoreCoord(h.MX, h.grid.y - 1);
    }

    // the gather (XGather3): the TP rings, simple placement in the mover column
    h.klt = e.KT / G;
    h.sps = h.klt / e.KB;
    h.rp = std::max<uint32_t>(1, std::min<uint32_t>(e.KB, h.payload / 2048));
    for (uint32_t r = 0; r < h.R; ++r) {
        for (uint32_t p = 0; p < G; ++p) {
            GatherChip ch;
            ch.rg.p = p;
            if (h.ring || p < G - 1) {
                ch.rg.next = Coord{r, u32(pmod(p + 1, G))};
            }
            if (h.ring || p > 0) {
                ch.rg.prev = Coord{r, u32(pmod(int64_t(p) - 1, G))};
            }
            auto [f, b] = gather_schedule(p, G, h.ring, h.ring);
            ch.rg.sched = {f, b};
            h.chips[{r, p}] = ch;
        }
    }
    for (auto& [coord, ch] : h.chips) {
        for (XDir d : {XDir::Fwd, XDir::Bwd}) {
            const auto& peer = ch.rg.peer(d);
            if (!peer) {
                continue;
            }
            const bool needs_ready = !h.chips.at(*peer).rg.sched[static_cast<int>(opp(d))].empty();
            if (ch.rg.sched[static_cast<int>(d)].empty() && !needs_ready) {
                continue;
            }
            auto links = tt::tt_fabric::get_forwarding_link_indices(mesh, h.node(coord), h.node(*peer));
            TT_FATAL(
                links.size() >= h.L,
                "fused_head: ({}, {}) -> ({}, {}) has {} usable link(s), {} requested",
                coord.first,
                coord.second,
                peer->first,
                peer->second,
                links.size(),
                h.L);
            links.resize(h.L);
            ch.rg.links[static_cast<int>(d)] = links;
        }
    }
    {
        std::set<std::pair<size_t, size_t>> allowed;
        for (uint32_t y = 0; y < h.grid.y; ++y) {
            allowed.insert({h.MX, y});
        }
        for (auto& [coord, ch] : h.chips) {
            std::set<std::pair<size_t, size_t>> taken;
            auto below = [&]() {  // (_worker_below at NoC column 0: the free cell of the lowest virtual y)
                std::optional<CoreCoord> best;
                for (const auto& [x, y] : allowed) {
                    if (taken.count({x, y})) {
                        continue;
                    }
                    const CoreCoord c(x, y);
                    if (!best || h.virt(c).y < h.virt(*best).y) {
                        best = c;
                    }
                }
                TT_FATAL(best.has_value(), "fused_head: the mover column has too few cores for the ports");
                taken.insert({best->x, best->y});
                return *best;
            };
            for (XDir d : {XDir::Fwd, XDir::Bwd}) {
                for (uint32_t l = 0; l < h.L; ++l) {
                    ch.ports.emplace_back(d, l, below());
                }
            }
            std::vector<uint32_t> ys;
            for (uint32_t y = 0; y < h.grid.y; ++y) {
                ys.push_back(y);
            }
            const uint32_t mid = ys[ys.size() / 2];
            std::vector<uint32_t> walk;
            for (uint32_t y : ys) {
                if (y >= mid) {
                    walk.push_back(y);
                }
            }
            for (uint32_t y : ys) {
                if (y < mid) {
                    walk.push_back(y);
                }
            }
            for (uint32_t l = 0; l < h.L; ++l) {
                for (uint32_t y : walk) {
                    if (!taken.count({h.MX, y})) {
                        ch.copy.emplace_back(h.MX, y);
                        taken.insert({h.MX, y});
                        break;
                    }
                }
            }
            TT_FATAL(ch.copy.size() == h.L, "fused_head: the mover column has too few cores for the copy cores");
        }
    }
    // (the ring's far side: the way the fabric's routing sends a packet - XGather3.ring_dir)
    if (h.ring && G % 2 == 0) {
        for (const auto& [coord, ch] : h.chips) {
            const auto fd = tt::tt_fabric::get_eth_forwarding_direction(h.node(coord), h.node(*ch.rg.next));
            for (const auto& [qc, qch] : h.chips) {
                if (qc.first != coord.first || pmod(int64_t(qch.rg.p) - ch.rg.p, G) * 2 != G) {
                    continue;
                }
                const XDir d = tt::tt_fabric::get_eth_forwarding_direction(h.node(coord), h.node(qc)) == fd ? XDir::Fwd
                                                                                                            : XDir::Bwd;
                auto [it, fresh] = h.ties.emplace(std::make_pair(ch.rg.p, qch.rg.p), d);
                TT_FATAL(fresh || it->second == d, "fused_head: ring ties differ between mesh rows");
            }
        }
    }

    // the arena: the engine's, grown to what the router cores need
    h.arena_bytes = e.arena_bytes;
    const int64_t need = al64(router_min_bytes(G, sh.E, sh.K));
    if (h.arena_bytes < need && need <= e.args.arena_max) {
        h.arena_bytes = need;
    }

    // the mailbox words (HeadSE3)
    h.HS = config.handoff_bytes;
    const auto rt = routing_paths(h.R, G, h.ring, h.wrap_y, h.L);
    h.fslots = std::max<uint32_t>(16, std::holds_alternative<RoutingPaths>(rt) ? std::get<RoutingPaths>(rt).nf : 16);
    h.lw_off = (2 * GX * GY * h.HS / 4 + 4 * 8 + 4 * h.fslots + 4 + 3) / 4 * 4;
    const uint32_t nrc = nrt;
    h.port_words = std::max<uint32_t>(16, (G - 1) * u32(cdiv(nrc, h.L)));
    h.zw_off = h.lw_off + 4 * h.port_words;
    h.mbox_words = h.zw_off + 16;

    // the router cores (RouterStream)
    h.RB = sh.rows / 32 / G;
    h.KT = sh.H / G / 32;
    h.ET = sh.E / 32;
    h.K = sh.K;
    TT_FATAL(h.rcores.size() >= h.RB, "fused_head: {} router cores < {}", h.rcores.size(), h.RB);
    h.rcores.resize(h.RB);
    {
        std::tie(h.roff, h.rbytes) = router_layout(G, h.ET, h.K, h.ET);
        TT_FATAL(
            h.rbytes <= h.arena_bytes, "fused_head: router cores need {} B of the arena's {}", h.rbytes, h.arena_bytes);
        h.NS = u32(cdiv(h.rcores.size(), h.L));
        h.pbuf_sub = int64_t(G - 1) * h.NS * h.ET * 4096;
    }

    // the routing exchange (RoutingStream)
    h.S = sh.rows / G;
    h.EPC = sh.EPC();
    h.A = sh.A;
    h.rows_cap = u32(routing_worst_rows(h.N, h.S, h.K, h.EPC));
    {
        const uint32_t nc = u32(h.rcores.size());
        RoutingSizes sz{};
        int64_t step = 0, pl0 = 0;
        uint32_t M = std::min({2 * nc, h.EPC, h.A, 14u});
        bool ok = false;
        for (; M > 0; --M) {
            sz = routing_a2a_sizes(h.N, h.S, h.K, h.EPC, h.A, h.rows_cap, M);
            step = al64(sz.plan_l1);
            const int64_t per = cdiv(M, nc);
            pl0 = per * step <= std::get<1>(*std::find_if(
                                    h.roff.begin(), h.roff.end(), [](const auto& r) { return r.first == "bias"; }))
                      ? 0
                      : al64(h.rbytes);
            if (sz.fits && pl0 + per * step <= h.arena_bytes) {
                ok = true;
                break;
            }
        }
        TT_FATAL(ok, "fused_head: routing planner: no room ({})", sz.why);
        h.M = M;
        h.inst_cap = u32(sz.inst_cap);
        h.plan_l1 = sz.plan_l1;
        h.pl_step = step;
        for (uint32_t q = 0; q < M; ++q) {
            h.pbases_off.push_back(pl0 + int64_t(q / nc) * step);
        }
    }
    h.HBP = routing_header_bytes(h.EPC, h.M);
    h.BB = al16(2 * int64_t(h.EPC));
    h.BBP = al16(2 * (int64_t(h.EPC) + 1));
    h.PCAP = routing_slot_bytes(h.S, h.K, h.EPC, h.M);
    h.MAXP = int64_t(h.payload) / 64 * 64;
    h.NF = h.fslots;
    for (const auto& [coord, _] : h.chips) {
        h.serves[coord.first * h.C + coord.second] = h.serve(coord);
    }
    for (const auto& [coord, _] : h.chips) {  // (ra_y_send waits only when it sends: no hello may land elsewhere)
        TT_FATAL(
            h.y_hello_to(coord) || !h.y_hellos_in(coord),
            "fused_head: chip ({}, {}): a Y hello lands on a chip that sends none",
            coord.first,
            coord.second);
    }
    for (const auto& [_, sv] : h.serves) {
        for (const auto& [k, v] : sv) {
            (k.y ? h.ny : h.nx) = std::max<uint32_t>(k.y ? h.ny : h.nx, u32(v.size()));
        }
    }
    h.xbase_off = al64(h.pbuf_sub);
    h.fbase_off = h.xbase_off + int64_t(h.nx) * h.PCAP;
    for (uint32_t s_ = 0; s_ < h.N; ++s_) {
        for (uint32_t d = 0; d < h.N; ++d) {
            if (d == s_) {
                continue;
            }
            const auto steps = h.path(s_, d);
            for (uint32_t i = 1; i < steps.size(); ++i) {
                auto& lst = h.fwd[steps[i]];
                h.fwd_at[{s_, d, i}] = u32(lst.size());
                lst.emplace_back(s_, d, i);
            }
        }
    }
    {
        std::set<SendKey> ks;
        for (const auto& [k, _] : h.fwd) {
            ks.insert(k.second);
        }
        h.fkeys.assign(ks.begin(), ks.end());
    }
    uint32_t nf = 0;
    for (const auto& [_, v] : h.fwd) {
        nf = std::max(nf, u32(v.size()));
    }
    TT_FATAL(
        h.xbase_off + int64_t(h.nx + nf) * h.PCAP <= h.arena_bytes, "fused_head: ports: no L1 for the routing buffers");
    // the x-gather's packet staging (kXgCb): an arena region on the port cores, after their routing buffers (the
    // ports hold nothing else in the arena: logits partials [0, pbuf_sub), routing [xbase_off, + (nx + nf) PCAP))
    h.xg_bytes = 2 * int64_t(h.group) * h.rp * 2048;
    h.xg_off = al64(h.xbase_off + int64_t(h.nx + nf) * h.PCAP);
    if (h.arena_bytes < h.xg_off + h.xg_bytes && h.xg_off + h.xg_bytes <= e.args.arena_max) {
        h.arena_bytes = al64(h.xg_off + h.xg_bytes);
    }
    TT_FATAL(
        h.xg_off + h.xg_bytes <= h.arena_bytes,
        "fused_head: ports: no L1 for the x-gather staging ({} B at {} of the arena's {})",
        h.xg_bytes,
        h.xg_off,
        h.arena_bytes);
    TT_FATAL(2 * int64_t(h.ny) * h.PCAP <= h.arena_bytes, "fused_head: the watcher core: no L1 for the Y buffers");
    return m;
}

std::vector<HeadWordsSpec> head_words(const HeadModel& h) {
    const auto ports = cells(h.all_ports());
    auto with = [](const std::vector<CoreCoord>& a, const std::vector<CoreCoord>& b) {
        std::vector<CoreCoord> v = a;
        for (const auto& c : b) {
            if (std::find(v.begin(), v.end(), c) == v.end()) {
                v.push_back(c);
            }
        }
        return v;
    };
    const auto pw = with(h.all_ports(), {h.watcher});
    std::vector<HeadWordsSpec> out = {
        {"mbox", ports, h.mbox_words},
        {"arr", h.engine_cores(), 4 * 4},
        {"xg", cells(pw), 4 * (3 + 2 * h.L)},
        {"rt", cells(h.rcores), 4 * 2},
        {"slot_words", cells(h.rcores), 4 * std::max<uint32_t>(1, h.G - 1)},
        {"sfree", cells(h.rcores), 32},
        {"ra_c", cells({h.rcores[0]}), 4 * 2},
        {"ra_p", cells(h.rcores), 4 * 9},
        {"ra_s", cells(pw), 4 * 4},
        {"yhello", cells({h.watcher}), 16},
    };
    out.push_back({"ewords", h.engine_cores(), 64});
    return out;
}

namespace head_program_detail {

// A runtime arg: a value, and for an address the source it points into (HeadSrc) and the offset there.
struct Arg {
    uint32_t v = 0;
    int32_t src = -1;
    uint32_t off = 0;
    Arg() = default;
    Arg(uint32_t value) : v(value) {}  // NOLINT(google-explicit-constructor): plain values read as args
    Arg(uint32_t value, int32_t source, uint32_t offset) : v(value), src(source), off(offset) {}
    bool operator==(const Arg&) const = default;
};
// an address arg further into its source
inline Arg operator+(const Arg& a, uint32_t d) { return Arg(a.v + d, a.src, a.src >= 0 ? a.off + d : 0); }
using Args = std::vector<Arg>;

void add(Args& a, uint32_t v) { a.emplace_back(v); }
void add(Args& a, const Arg& v) { a.push_back(v); }
void add(Args& a, const std::vector<uint32_t>& v) {
    for (uint32_t x : v) {
        a.emplace_back(x);
    }
}
void add(Args& a, const Args& v) { a.insert(a.end(), v.begin(), v.end()); }

std::vector<uint32_t> values(const Args& a) {
    std::vector<uint32_t> out;
    out.reserve(a.size());
    for (const auto& x : a) {
        out.push_back(x.v);
    }
    return out;
}

// se_engine3._common_span: (per-core lists without [start, start + n), the common args) when every core holds the
// same span; else (the full lists, [0, 0])
std::pair<std::vector<std::pair<CoreCoord, Args>>, Args> common_span(
    const std::vector<std::pair<CoreCoord, Args>>& lists, size_t start, size_t n) {
    bool shared = n > 0;
    for (const auto& [_, a] : lists) {
        shared = shared && a.size() >= start + n &&
                 std::equal(a.begin() + start, a.begin() + start + n, lists[0].second.begin() + start);
    }
    if (!shared) {
        return {lists, Args{0u, 0u}};
    }
    std::vector<std::pair<CoreCoord, Args>> per;
    for (const auto& [c, a] : lists) {
        Args b(a.begin(), a.begin() + start);
        b.insert(b.end(), a.begin() + start + n, a.end());
        per.emplace_back(c, b);
    }
    Args common = {u32(start), u32(n)};
    common.insert(common.end(), lists[0].second.begin() + start, lists[0].second.begin() + start + n);
    return {per, common};
}

tt::DataFormat w_format(tt::tt_metal::DataType t) {
    switch (t) {
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        default: return tt::DataFormat::Bfp8_b;
    }
}

// se_engine3.glu_defines: (SE_GLU code, its constants) of the shared expert's activation
std::pair<uint32_t, std::vector<std::pair<std::string, float>>> glu_defines(const ExpertActivation& act) {
    uint32_t code = 1;
    float limit = 10.0f;
    switch (act.kind) {
        case ExpertActivationKind::Silu: code = 1; break;
        case ExpertActivationKind::Situ: code = 2; break;
        case ExpertActivationKind::ClampedSilu: code = 3; break;
        case ExpertActivationKind::SwigluOai:
            code = 4;
            limit = 7.0f;
            break;
    }
    return {
        code,
        {{"SE_GLU_BETA", act.beta.value_or(4.0f)},
         {"SE_GLU_LBETA", act.linear_beta.value_or(25.0f)},
         {"SE_GLU_LIMIT", act.limit.value_or(limit)},
         {"SE_GLU_ALPHA", act.alpha.value_or(1.702f)}}};
}

}  // namespace head_program_detail

// One chip's program (HeadSE3._run's loop body).
class ChipProgram {
public:
    ChipProgram(const HeadModel& m, const HeadBuffers& b, Coord coord) :
        h(m), e(*m.eng), rs(*m.rs), bufs(b), coord(coord) {
        p = h.pos(coord);
        me = coord.first * h.C + coord.second;
        base = S(SRC_ARENA);
        mbox = word("mbox");
        for (uint32_t i = 0; i < (e.NCH > 1 ? 4u : 3u); ++i) {
            arr.push_back(word("arr") + 16 * i);
        }
        zfree = mbox + 4 * h.zw_off;
        const Arg xg = word("xg");
        arrival = xg;
        ready = xg + 16;
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t v = 0; v < h.L; ++v) {
                watch[{static_cast<XDir>(i), v}] = xg + 16 * (2 + i * h.L + v);
            }
        }
        xloaded = xg + 16 * (2 + 2 * h.L);
        rt_arr0 = word("slot_words");
        rt_arr1 = word("rt") + 16;
        sfree = word("sfree");
        const std::vector<std::string> pk = {"calls_p", "calls_p2", "go", "go2", "parts", "ex", "ex2", "rows", "rows2"};
        for (uint32_t i = 0; i < pk.size(); ++i) {
            ra[pk[i]] = word("ra_p") + 16 * i;
        }
        ra["c0"] = word("ra_c");
        ra["c1"] = word("ra_c") + 16;
        const std::vector<std::string> sk = {"calls_s", "calls_s2", "ready", "ready2"};
        for (uint32_t i = 0; i < sk.size(); ++i) {
            ra[sk[i]] = word("ra_s") + 16 * i;
        }
        for (const auto& [d, l, c] : h.chip(coord).ports) {
            pc[{d, l}] = h.virt(c);
        }
        sv = h.serves.at(me);
    }

    ProgramDescriptor build(HeadPatches* patches);

private:
    const HeadModel& h;
    const SeEngineSizing& e;
    const RsPlan& rs;
    const HeadBuffers& bufs;
    Coord coord;
    uint32_t p = 0, me = 0;
    Arg base, mbox, zfree, arrival, ready, xloaded, rt_arr0, rt_arr1, sfree;
    std::vector<Arg> arr;
    std::map<std::pair<XDir, uint32_t>, Arg> watch;
    std::map<std::string, Arg> ra;
    // the address args of each kernel: by its (local) descriptor's address while it is built, then - push moves it into
    // its function's kernel list - in pushed, one entry per kernel in list order (a descriptor's address does not
    // survive the moves; take_pushed hands a list's entries to the final assembly)
    std::map<const KernelDescriptor*, std::vector<HeadPatches::Arg>> kpatches;
    std::vector<std::vector<HeadPatches::Arg>> pushed;
    void push(std::vector<KernelDescriptor>& ks, KernelDescriptor& k) {
        auto it = kpatches.find(&k);
        pushed.push_back(it == kpatches.end() ? std::vector<HeadPatches::Arg>{} : std::move(it->second));
        if (it != kpatches.end()) {
            kpatches.erase(it);
        }
        ks.push_back(std::move(k));
    }
    std::vector<std::vector<HeadPatches::Arg>> take_pushed() { return std::exchange(pushed, {}); }

    // a source's address arg (0 for an absent tensor)
    Arg S(HeadSrc s) const { return Arg(bufs.addresses[s], bufs.addresses[s] ? int32_t(s) : -1, 0); }
    // a core's / the common args of a kernel, their addresses recorded
    void set_rt(KernelDescriptor& k, const CoreCoord& core, const Args& a) {
        for (uint32_t i = 0; i < a.size(); ++i) {
            if (a[i].src >= 0) {
                kpatches[&k].push_back({0, false, core, i, uint16_t(a[i].src), a[i].off});
            }
        }
        k.runtime_args.emplace_back(core, values(a));
    }
    void set_common(KernelDescriptor& k, const Args& a) {
        for (uint32_t i = 0; i < a.size(); ++i) {
            if (a[i].src >= 0) {
                kpatches[&k].push_back({0, true, CoreCoord{}, i, uint16_t(a[i].src), a[i].off});
            }
        }
        k.common_runtime_args = values(a);
    }
    std::map<std::pair<XDir, uint32_t>, CoreCoord> pc;
    std::map<SendKey, std::vector<uint32_t>> sv;
    ProgramDescriptor desc;

    Arg word(const std::string& n) const {
        const uint32_t i = h.word_index(n);
        return Arg(bufs.addresses[SRC_WORD0 + i], int32_t(SRC_WORD0 + i), 0);
    }
    Arg A(const std::string& region) const { return base + u32(e.at(region)); }
    Arg RA(const std::string& region) const { return base + u32(region_of_router(region)); }
    int64_t region_of_router(const std::string& k) const {
        for (const auto& [n, v] : h.roff) {
            if (n == k) {
                return v;
            }
        }
        TT_THROW("fused_head: no router region {}", k);
    }
    CBDescriptor arena_cb(
        uint8_t idx, int64_t off, uint32_t size, tt::DataFormat fmt, uint32_t page, const CoreRangeSet& cores) const {
        CBDescriptor c;
        c.total_size = size;
        c.core_ranges = cores;
        c.format_descriptors = {CBFormatDescriptor{.buffer_index = idx, .data_format = fmt, .page_size = page}};
        c.buffer = bufs[SRC_ARENA];
        c.address_offset = u32(off);
        return c;
    }
    uint32_t port_xy(XDir d, uint32_t l) const {
        const auto& c = pc.at({d, l});
        return u32(c.x | c.y << 16);
    }

    // the pieces
    std::vector<uint32_t> sor() const;
    std::vector<KernelDescriptor> engine(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> gather(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> router(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> y_kernels();
    Args rs_brisc(uint32_t cx, uint32_t cy) const;
    Args la_brisc(uint32_t cx, uint32_t cy) const;
    Args la_ncrisc(uint32_t cy) const;
    Args rs_ncrisc(uint32_t cx, uint32_t cy) const;
    Args prep_ncrisc(uint32_t cx, uint32_t cy) const;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> prep_units(
        uint32_t cx, uint32_t cy) const;
    std::vector<uint32_t> prep_compute(const std::vector<uint32_t>& ra_, size_t n_pre, uint32_t cx, uint32_t cy) const;
    std::vector<Args> engine_dests() const;
    Args rs_block(XDir d, uint32_t v) const;
    Args la_block(XDir d, uint32_t v) const;
    Args lg_block(XDir d, uint32_t v) const;
    Args port_block(XDir d, uint32_t v) const;
    Args send_block(const SendKey& k) const;
    Args fwd_block(const SendKey& k) const;
    std::vector<uint32_t> next_hop(uint32_t s, uint32_t d, uint32_t i) const;
    Args y_args(const SendKey& k, const CoreCoord& core);
    void routing_defines(KernelDescriptor& k) const;
    // The routing kernels' DRAM tensors (ra_accessors.hpp): their accessor args after the kernel's own compile-time
    // args, from RA_ACC_CT on - plan, counts, regions, token index, table, slots, forwarding slots, the padded chunk's
    // start / end scalars (an interleaved DRAM placeholder for one this call has none of).
    void routing_accessors(KernelDescriptor& k) const {
        define(k, "RA_ACC_CT", iv(k.compile_time_args.size()));
        for (HeadSrc s :
             {SRC_PLAN,
              SRC_COUNTS,
              SRC_REGIONS,
              SRC_TOKEN_INDEX,
              SRC_TABLE,
              SRC_SLOTS,
              SRC_FSLOTS,
              SRC_START,
              SRC_END}) {
            (bufs[s] ? tt::tt_metal::TensorAccessorArgs(*bufs[s])
                     : tt::tt_metal::TensorAccessorArgs::create_dram_interleaved())
                .append_to(k.compile_time_args);
        }
    }
    void planner_defines(KernelDescriptor& k, uint32_t ct0) const;
    std::function<Args(uint32_t)> planner_args() const;
    Args ra_args(uint32_t j) const;
    Args ra_args2(uint32_t j) const;
    std::vector<uint32_t> fabric(const Coord& peer, uint32_t link, const CoreCoord& core) {
        std::vector<uint32_t> out;
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            h.node(coord), h.node(peer), link, desc, core, out);
        return out;
    }
};

std::vector<uint32_t> ChipProgram::sor() const {
    std::vector<uint32_t> so = h.cfg.lay.xdram ? std::vector<uint32_t>{0} : h.shard_order(coord).first;
    std::vector<uint32_t> out = {e.NSG / u32(so.size()), u32(so.size())};
    out.insert(out.end(), so.begin(), so.end());
    return out;
}

// ---- se_engine3.SeEngine3.pieces ----

Args ChipProgram::rs_brisc(uint32_t cx, uint32_t cy) const {
    const uint32_t k = rs.k(cx, cy);
    const auto hs = rs.handoffs(p, cx, cy);
    Args out = {u32(hs.size()), mbox + h.HS * k};
    for (const auto& [_, d] : hs) {
        const auto& c = pc.at({d == RsDir::Fwd ? XDir::Fwd : XDir::Bwd, rs.link(k)});
        out.emplace_back(u32(c.x));
        out.emplace_back(u32(c.y));
    }
    return out;
}

Args ChipProgram::la_brisc(uint32_t cx, uint32_t cy) const {
    if (!e.LT) {
        return {};
    }
    const uint32_t k = cy * e.GX + cx;
    std::vector<uint32_t> dst;
    for (uint32_t b = 0; b < h.G; ++b) {
        if (b != p && e.la_rows(cy, b).second) {
            dst.push_back(b);
        }
    }
    const Arg word_ = mbox + h.HS * e.GX * e.GY + 128;
    Args out = {u32(dst.size()), word_ + h.HS * k};
    for (uint32_t b : dst) {
        const XDir d = rs.kind == RsKind::Line ? (b > p ? XDir::Fwd : XDir::Bwd) : h.ring_dir(p, b);
        const auto& c = pc.at({d, k % rs.nv});
        out.emplace_back(u32(c.x));
        out.emplace_back(u32(c.y));
    }
    return out;
}

Args ChipProgram::la_ncrisc(uint32_t cy) const {
    if (!e.LT) {
        return {};
    }
    const uint32_t nr = e.la_rows(cy, p).second;
    return {nr, nr ? h.G - 1 : 0u, arr[2]};
}

Args ChipProgram::rs_ncrisc(uint32_t cx, uint32_t cy) const {
    const auto a = rs.arrivals(p, cx, cy);
    const auto zf = rs.zone_slots(p, cx, cy, RsDir::Fwd), zb = rs.zone_slots(p, cx, cy, RsDir::Bwd);
    auto [y0, stride, w] = rs.fin_map(p, cx, cy);
    Args out = {u32(a[0].size())};
    for (const auto& x : a[0]) {
        out.push_back(x.threshold | zf[x.slot].second << 16);
    }
    out.push_back(u32(a[1].size()));
    for (const auto& x : a[1]) {
        out.push_back(x.threshold | zb[x.slot].second << 16);
    }
    for (const Arg& v : std::initializer_list<Arg>{rs.count(cx, cy, p), y0, stride, w, arr[0], arr[1]}) {
        out.push_back(v);
    }
    if (e.NCH == 1) {
        return out;
    }
    std::vector<uint32_t> y0s;
    if (rs.scatter == -1) {
        if (rs.count(cx, cy, p)) {
            out[out.size() - 5] = y0 + (e.grow(0, cy * e.MT) - cy * e.MT) * rs.NB;
        }
        for (uint32_t m = 1; m < e.NCH; ++m) {
            y0s.push_back(y0 + (e.grow(m, cy * e.MT) - cy * e.MT) * rs.NB);
        }
    } else {
        for (uint32_t m = 1; m < e.NCH; ++m) {
            y0s.push_back(y0 + m * rs.RB * rs.NTH);
        }
    }
    const uint32_t link = rs.link(rs.k(cx, cy));
    auto facing = [&](int d) -> uint32_t {
        if (a[d].empty()) {
            return 0;
        }
        const auto& c = pc.at({d == 0 ? XDir::Bwd : XDir::Fwd, link});
        return u32(c.x | c.y << 16);
    };
    out.insert(out.end(), y0s.begin(), y0s.end());
    for (const Arg& v : std::initializer_list<Arg>{
             facing(0), facing(1), zfree, arr.size() > 3 ? arr[3] : Arg(0u), u32(rs.finb_readers(p, cx, cy).size())}) {
        out.push_back(v);
    }
    if (e.fin_rows()) {
        const uint32_t col = y0 - cy * e.MT * rs.NB;
        for (uint32_t m = 0; m < e.NCH; ++m) {
            for (uint32_t r = 0; r < e.MT; ++r) {
                out.push_back(col + e.grow(m, cy * e.MT + r) * rs.NB);
            }
        }
    }
    return out;
}

std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> ChipProgram::prep_units(
    uint32_t cx, uint32_t cy) const {
    const uint32_t MT = e.MT, KB = e.KB, G = e.shards;
    const uint32_t RBo = e.rows / 32 / G;
    const uint32_t row0 = me * RBo * 32;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> out;
    if (e.LT) {
        const auto [r0, nr] = e.la_rows(cy, p);
        const uint32_t lz = u32(e.at("latz") - e.at("lat"));
        const uint32_t rbc = e.CR / G;
        for (uint32_t m = 0; m < e.NCH; ++m) {
            const uint32_t cm = u32(m * e.LATP * 2048);
            const uint32_t lorow = cy * MT + r0 - p * rbc + m * rbc;
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t rr = (r0 + i) * e.LWMAX * 2048, n = e.lw[cx];
                out.emplace_back(cm + rr, n, 0, row0 + (lorow + i) * 32, (p * e.LT + e.l0[cx]) * 64);
                for (uint32_t sl = 0; sl < G - 1; ++sl) {
                    const uint32_t q = sl < p ? sl : sl + 1;
                    out.emplace_back(
                        cm + lz + sl * MT * e.LWMAX * 2048 + rr,
                        n,
                        1,
                        row0 + (lorow + i) * 32,
                        (q * e.LT + e.l0[cx]) * 64);
                }
            }
        }
        return out;
    }
    const std::vector<uint32_t> so = h.cfg.lay.xdram ? std::vector<uint32_t>{0} : h.shard_order(coord).first;
    const uint32_t sps = e.NSG / u32(so.size());
    std::map<uint32_t, uint32_t> rank;
    for (uint32_t k = 0; k < so.size(); ++k) {
        rank[so[k]] = k;
    }
    const uint32_t nf = e.NSG / 2;
    const uint32_t n0 = e.in_fam0(cx) ? nf / e.S0 : 0;
    const uint32_t nown = n0 + (e.in_fam1(cx) ? nf / e.S1 : 0);
    const uint32_t rbc = e.CR / G;
    for (uint32_t u = 0; u < RBo; ++u) {
        uint32_t m, lr;
        if (e.NCH > 1) {
            m = u / rbc;
            lr = p * rbc + u % rbc;
        } else {
            m = (p * RBo + u) / e.CR;
            lr = (p * RBo + u) % e.CR;
        }
        const uint32_t Rr = p * RBo + u;
        if (lr / MT != cy) {
            continue;
        }
        const uint32_t r = lr % MT;
        for (uint32_t j = 0; j < so.size() * sps; ++j) {
            const uint32_t pos = rank.at(j / sps) * sps + j % sps;
            const auto [col, f, _] = e.owner(pos);
            if (col != cx) {
                continue;
            }
            const uint32_t hh = pos / 2;
            const uint32_t slot = f ? n0 + hh / e.S1 : hh / e.S0;
            out.emplace_back(
                (((m % e.XR) * nown + slot) * MT + r) * KB * 2048, KB, 0, row0 + (Rr - p * RBo) * 32, j * KB * 64);
        }
    }
    return out;
}

std::vector<uint32_t> ChipProgram::prep_compute(
    const std::vector<uint32_t>& ra_, size_t n_pre, uint32_t cx, uint32_t cy) const {
    const uint32_t nops = ra_[0];
    std::vector<uint32_t> ops(ra_.begin() + 1, ra_.begin() + 1 + nops), rest(ra_.begin() + 1 + nops, ra_.end());
    uint32_t last = 0;
    for (uint32_t i = 0; i < ops.size(); ++i) {
        if ((ops[i] & 0xFF) == 0) {
            last = i;
        }
    }
    const uint32_t b = u32(n_pre + 1 + nops + 1 + rest.size());
    const auto us = prep_units(cx, cy);
    ops.insert(ops.begin() + last + 1, 3 | (b << 8));
    TT_FATAL(us.size() % e.NCH == 0, "fused_head: dispatch prep units not whole chunks");
    std::vector<uint32_t> out = {nops + 1};
    out.insert(out.end(), ops.begin(), ops.end());
    out.insert(out.end(), rest.begin(), rest.end());
    out.push_back(u32(us.size() / e.NCH));
    for (const auto& [o, n, lat, _r, _c] : us) {
        out.push_back(o);
        out.push_back(n | (lat << 16));
    }
    return out;
}

Args ChipProgram::prep_ncrisc(uint32_t cx, uint32_t cy) const {
    const auto us = prep_units(cx, cy);
    Args out = {u32(us.size()), S(SRC_XBUF)};
    for (const auto& [_o, n, _l, row, co] : us) {
        add(out, Args{row, co, n * 64});
    }
    return out;
}

// router_stream.RouterStream.engine_args: per row tile R its destinations (NoC x | y << 16, base, count address)
std::vector<Args> ChipProgram::engine_dests() const {
    std::vector<Args> dests;
    for (uint32_t Rr = 0; Rr < h.G * h.RB; ++Rr) {
        const uint32_t bb = Rr / h.RB, j = Rr % h.RB;
        Args dr;
        if (bb == p) {
            const auto c = h.virt(h.rcores[j]);
            dr.push_back(u32(c.x | c.y << 16));
            dr.push_back(RA("hold") + bb * h.ET * 4096);
            dr.push_back(rt_arr1);
        } else {
            const uint32_t l = j % h.L;
            const auto js = h.port_layout(l);
            const XDir d = h.rdir(p, bb);
            const auto ds = h.dests(p, d);
            const uint32_t k = u32(std::find(ds.begin(), ds.end(), bb) - ds.begin()) * u32(js.size()) +
                               u32(std::find(js.begin(), js.end(), j) - js.begin());
            const auto& c = pc.at({d, l});
            dr.push_back(u32(c.x | c.y << 16));
            dr.push_back(base + k * h.ET * 4096);
            dr.push_back(mbox + 4 * h.lw_off + 16 * k);
        }
        dests.push_back(dr);
    }
    return dests;
}

std::vector<KernelDescriptor> ChipProgram::engine(std::vector<CBDescriptor>& cbs) {
    const uint32_t GX = e.GX, GY = e.GY, MT = e.MT;
    const auto allc = h.engine_cores();
    const bool xdram = h.cfg.lay.xdram;
    Buffer* x = xdram ? bufs[SRC_X_FULL] : bufs[SRC_X];
    const Arg xa = S(xdram ? SRC_X_FULL : SRC_X);
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;
    auto cb = [&](uint8_t i, const std::string& key, int64_t size, tt::DataFormat dt, uint32_t page) {
        cbs.push_back(arena_cb(i, e.at(key), u32(size), dt, page, allc));
    };
    cb(XIN, "xin", 2 * MT * e.KB * 2048, bf16, 2048);
    cb(WIN, "win", kNws * e.WSLOT * e.wtb, w_format(h.cfg.w_dtype), e.wtb);
    cb(HOWN, "hown", MT * e.POMAX * 2048, bf16, 2048);
    cb(HALL, "hall", e.HSL * GX * MT * e.POMAX * 2048, bf16, 2048);
    auto rsz = [&](const std::string& k) {
        for (const auto& [n, v] : e.rs_sizes) {
            if (n == k) {
                return v;
            }
        }
        TT_THROW("fused_head: no RS region {}", k);
    };
    cb(kRsHold, "hold", rsz("hold"), fp32, 4096);
    cb(kRsFinb, "finb", rsz("finb"), bf16, 2048);
    cb(kRsArrf, "arrf", rsz("arrf"), bf16, 2048);
    cb(kRsArrb, "arrb", rsz("arrb"), bf16, 2048);
    cb(kRsOutf, "outf", rsz("outf"), bf16, 2048);
    cb(kRsFinc, "finc", rsz("finc"), bf16, 16);
    cb(kRsPsync, "psync", rsz("psync"), bf16, 16);
    if (e.NCH > 1) {
        cb(kRsFfree, "ffree", rsz("ffree"), bf16, 16);
    }
    if (e.LT) {
        cb(kLatCb, "lat", e.NCH * e.LATP * 2048, bf16, 2048);
    }
    {  // dispatch prep (always: the head plans it with the routing exchange)
        const std::string pk = e.LT ? "lat" : "xown";
        const int64_t pend =
            e.LT ? e.at("latz") + int64_t(e.shards - 1) * MT * e.LWMAX * 2048 + int64_t(e.NCH - 1) * e.LATP * 2048
                 : e.at("xown") + e.xown_t * 2048;
        cb(kPrepPu, pk, pend - e.at(pk), bf16, 2048);
        cb(kPrepPo, "po", 2 * e.PFULL * 2048, bf16, 2048);
        if (e.LT) {
            cb(kPrepLrdy, "lrdy", e.lrdy_bytes(), bf16, 16);
        }
    }
    if (e.xfree) {
        cb(kXfreeCb, "xfree", 16 * e.NCH, bf16, 16);
    }
    cb(kRtrRx, "rx", 2 * e.KC * MT * e.NCH * 2048, bf16, 2048);
    cb(kRtrRw, "rw", 2 * e.ETC * e.KC * 2048, bf16, 2048);
    cb(kRtrRo, "ro", e.ETC * MT * e.NCH * 4096, fp32, 4096);
    cb(ACC, "acc", e.acc_pages * 4096, fp32, 4096);

    std::vector<uint32_t> xa_ct, w_ct, y_ct;
    append_accessor(xa_ct, x);
    append_accessor(w_ct, bufs[SRC_W]);
    append_accessor(y_ct, bufs[SRC_Y]);
    const auto so = sor();
    // the gather's ports (NoC) to tell once my own-shard x is loaded, their counter, my XLD id, the "router done" word
    const Arg rdone = e.rtr_overlay ? zfree + 32 : Arg(0u);
    Args tail = {u32(h.chip(coord).ports.size()), xloaded};
    for (const auto& [d, l, c] : h.chip(coord).ports) {
        tail.push_back(u32(h.virt(c).x));
        tail.push_back(u32(h.virt(c).y));
    }
    tail.push_back(kXld);
    tail.push_back(rdone);
    const uint32_t nf = e.NSG / 2;
    std::vector<uint32_t> ftab = e.fam0;
    ftab.insert(ftab.end(), e.fam1.begin(), e.fam1.end());
    for (uint32_t c = 0; c < GX; ++c) {
        const uint32_t n0 = e.in_fam0(c) ? nf / e.S0 : 0;
        ftab.push_back(n0 + (e.in_fam1(c) ? nf / e.S1 : 0));
        ftab.push_back(n0);
    }
    const auto dests = engine_dests();
    const uint32_t nbank = u32(h.mesh->dram_grid_size().x);
    std::vector<std::pair<CoreCoord, Args>> bl, nl;
    size_t b_span0 = 0, b_spann = 0, n_span0 = 0, n_spann = 0;
    for (uint32_t c = 0; c < GX; ++c) {
        const uint32_t i0 = e.in_fam0(c) ? u32(std::find(e.fam0.begin(), e.fam0.end(), c) - e.fam0.begin()) : GX;
        const uint32_t i1 = e.in_fam1(c) ? u32(std::find(e.fam1.begin(), e.fam1.end(), c) - e.fam1.begin()) : GX;
        for (uint32_t cy = 0; cy < GY; ++cy) {
            Args b = {c, cy, e.po[c], e.h0[c], i0, i1, u32(h.evirt(c, cy).y)};
            add(b, xa);
            add(b,
                Args{
                    xdram ? e.KT : h.klt,
                    xdram ? 0u : p * h.klt,
                    A("xown"),
                    A("xin"),
                    A("hown"),
                    A("hall"),
                    u32(h.evirt(0, cy).x),
                    u32(h.evirt(GX - 1, cy).x)});
            add(b, S(SRC_X));
            add(b, A("rx"));
            add(b, ftab);
            for (uint32_t cc = 0; cc < GX; ++cc) {
                add(b, u32(h.evirt(cc, cy).x));
            }
            add(b, so);
            add(b, tail);
            add(b, S(SRC_Y));
            add(b, rs_brisc(c, cy));
            add(b, la_brisc(c, cy));
            Args rtr_a;
            // (se_engine3._rtr_args)
            const uint32_t e0 = c * e.ETC;
            const uint32_t nv = u32(std::max<int64_t>(0, std::min<int64_t>(e.ETC, int64_t(h.ET) - e0)));
            rtr_a = {
                S(SRC_X),
                S(SRC_RW),
                nv,
                e0,
                A("rx"),
                A("rw"),
                u32(h.evirt(GX - 1, cy).x),
                u32(h.evirt(0, cy).x),
                u32(h.evirt(0, cy).y),
                u32(nv > 0)};
            for (uint32_t rr = 0; rr < MT * e.NCH; ++rr) {
                add(rtr_a, dests[e.grow(rr / MT, cy * MT + rr % MT)]);
            }
            Args n = {
                cy,
                u32(h.evirt(c, cy).x),
                u32(((c * GY + cy) / nbank) * e.wown_t * e.wtb),
                e.n0d[c],
                (c * GY + cy) % nbank};
            add(n, rtr_a);
            add(n, e.wg[c]);
            add(n, e.wd[c]);
            add(n, Args{u32(h.evirt(c, 0).y), u32(h.evirt(c, GY - 1).y)});
            add(n, S(SRC_W));
            add(n, Args{A("wown"), A("win"), A("words")});
            add(n, S(SRC_Y));
            for (uint32_t yy = 0; yy < GY; ++yy) {
                add(n, u32(h.evirt(c, yy).y));
            }
            add(n, so);
            add(n, tail);
            add(n, rs_ncrisc(c, cy));
            add(n, la_ncrisc(cy));
            add(n, prep_ncrisc(c, cy));
            bl.emplace_back(CoreCoord(c, cy), b);
            nl.emplace_back(CoreCoord(c, cy), n);
            b_span0 = 7;
            b_spann = 9 + 2 + GX + so.size() + tail.size() + 1;
            n_span0 = 5 + rtr_a.size() + e.wg[c].size() + e.wd[c].size();
            n_spann = 7 + GY + so.size() + tail.size();
        }
    }
    auto [b_rt, b_crt] = common_span(bl, b_span0, b_spann);
    auto [n_rt, n_crt] = common_span(nl, n_span0, n_spann);
    const Arg ew = word("ewords");
    b_crt.emplace_back(ew);  // (SE_EW: common[2 + n])
    n_crt.emplace_back(ew);

    const std::vector<uint32_t> bct = {GX, MT, e.KB, e.NSG, e.NPG, e.NCH, e.POMAX, e.S0, e.S1, kXready};
    const std::vector<uint32_t> nct = {
        GY, MT, e.KB, e.NSG, e.NPG, e.NCH, e.KBD, e.NSD, e.NPD, kNws, e.WSLOT, e.wtb, kWv, kWfree};
    // the defines every engine kernel takes (`nd`)
    std::vector<std::pair<std::string, std::string>> nd = {
        {"SE_HSL", iv(e.HSL)}, {"SE_ACCP", uv(e.acc_pages)}, {"SE_DT", iv(kDestTiles)}};
    if (e.NCH > 1 && e.shards > 1) {
        nd.emplace_back("SE_GROW_OWN_CR", iv(e.CR / e.shards));
        nd.emplace_back("SE_GROW_OWN_RT", iv(e.rows / 32 / e.shards));
    } else {
        nd.emplace_back("SE_CR", iv(e.CR));  // (GROW's chunk-major form: row tiles a chunk)
    }
    if (e.WRING) {
        nd.emplace_back("SE_WRING", iv(e.WRING));
    }
    if (e.NCH > 1 && !e.xfree) {
        nd.emplace_back("SE_PREP_LAST", "");
    }
    if (e.xring) {
        nd.emplace_back("SE_XRING", uv(e.XR));
    }
    if (e.xfree) {
        nd.emplace_back("SE_XFREE", iv(kXfreeCb));
    }
    nd.emplace_back("SE_NENT", iv(rs.G));
    for (const auto& [k, v] : std::vector<std::pair<std::string, uint32_t>>{
             {"SE_HOLD", kRsHold},
             {"SE_FINB", kRsFinb},
             {"SE_ARRF", kRsArrf},
             {"SE_ARRB", kRsArrb},
             {"SE_FINC", kRsFinc},
             {"SE_OUTF", kRsOutf},
             {"SE_PSYNC", kRsPsync},
             {"SE_FFREE", kRsFfree}}) {
        nd.emplace_back(k, iv(v));
    }
    if (e.fin_rows()) {
        nd.emplace_back("SE_FIN_ROWS", "");
    }
    nd.emplace_back("SE_ZF", iv(rs.zone_tiles(RsDir::Fwd)));
    nd.emplace_back("SE_ZB", iv(rs.zone_tiles(RsDir::Bwd)));
    if (rs.pm) {
        nd.emplace_back("SE_RS_PM", uv(rs.npass));
    }
    if (e.LT) {
        nd.emplace_back("SE_LAT", iv(kLatCb));
        nd.emplace_back("SE_NGU", iv(e.NGU));
        nd.emplace_back("SE_LWMAX", iv(e.LWMAX));
        nd.emplace_back("SE_LATP", iv(e.LATP));
    }
    const size_t prep_ct0 = nct.size() + w_ct.size() + y_ct.size();
    const size_t nct_len = prep_ct0 + accessor_len(bufs[SRC_XBUF]);
    nd.emplace_back("SE_RX", iv(kRtrRx));
    nd.emplace_back("SE_RW", iv(kRtrRw));
    nd.emplace_back("SE_RO", iv(kRtrRo));
    nd.emplace_back("SE_KTO", iv(h.KT));
    nd.emplace_back("SE_ETC", iv(e.ETC));
    nd.emplace_back("SE_MTR", iv(MT * e.NCH));
    nd.emplace_back("SE_RG", iv(e.RG));
    nd.emplace_back("SE_RD", iv(3));
    nd.emplace_back("XACK_ID", iv(kXack));
    nd.emplace_back("WACK_ID", iv(kWack));
    nd.emplace_back("RXR_ID", iv(kRxr));
    nd.emplace_back("RWR_ID", iv(kRwr));
    if (e.rtr_overlay) {
        nd.emplace_back("SE_RDONE", "");
    }
    nd.emplace_back("RXS_ID", iv(kRxs));
    nd.emplace_back("SE_KC", iv(e.KC));
    if (!e.xring) {
        nd.emplace_back("SE_RTRX_L1", "");
        nd.emplace_back("XRL_ID", iv(kXrl));
    }
    nd.emplace_back("SE_RTR_BCT0", iv(bct.size() + xa_ct.size()));
    nd.emplace_back("SE_RTR_CT0", iv(nct_len));
    nd.emplace_back("SE_PU", iv(kPrepPu));
    nd.emplace_back("SE_PO", iv(kPrepPo));
    nd.emplace_back("SE_PREP_FULL", iv(e.PFULL));
    nd.emplace_back("SE_PREP_BCT", iv(e.PBCT));
    nd.emplace_back("SE_PREP_ROW", iv(bufs[SRC_XBUF]->aligned_page_size()));
    nd.emplace_back("SE_PREP_CT0", iv(prep_ct0));
    if (e.LT) {
        nd.emplace_back("SE_LRDY", iv(kPrepLrdy));
    }

    std::vector<KernelDescriptor> ks;
    auto brisc = kernel("dataflow/se_brisc.cpp", allc, dm(DataMovementProcessor::RISCV_0, NOC::NOC_0));
    brisc.compile_time_args = bct;
    brisc.compile_time_args.insert(brisc.compile_time_args.end(), xa_ct.begin(), xa_ct.end());
    append_accessor(brisc.compile_time_args, bufs[SRC_X]);
    brisc.defines = KernelDescriptor::Defines(nd.begin(), nd.end());
    for (const auto& [c, a] : b_rt) {
        set_rt(brisc, c, a);
    }
    set_common(brisc, b_crt);
    auto ncrisc = kernel("dataflow/se_ncrisc.cpp", allc, dm(DataMovementProcessor::RISCV_1, NOC::NOC_1));
    ncrisc.compile_time_args = nct;
    auto& nca = ncrisc.compile_time_args;
    nca.insert(nca.end(), w_ct.begin(), w_ct.end());
    nca.insert(nca.end(), y_ct.begin(), y_ct.end());
    append_accessor(nca, bufs[SRC_XBUF]);
    append_accessor(nca, bufs[SRC_X]);
    append_accessor(nca, bufs[SRC_RW]);
    ncrisc.defines = brisc.defines;
    for (const auto& [c, a] : n_rt) {
        set_rt(ncrisc, c, a);
    }
    set_common(ncrisc, n_crt);
    push(ks, brisc);
    push(ks, ncrisc);

    const auto [glu_code, glu_vals] = glu_defines(h.cfg.glu);
    std::vector<uint32_t> pre_base;
    for (const auto& [key, cols] : e.groups) {
        std::vector<CoreRange> rr;
        for (uint32_t c : cols) {
            rr.emplace_back(CoreCoord(c, 0), CoreCoord(c, GY - 1));
        }
        ComputeConfigDescriptor comp{
            .math_fidelity = h.cfg.fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = kDestTiles == 8,
            .math_approx_mode = false,
        };
        comp.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        comp.unpack_to_dest_mode[ACC] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        std::vector<uint32_t> pre = key.first;
        pre.insert(pre.end(), key.second.begin(), key.second.end());
        pre.insert(pre.end(), e.h0.begin(), e.h0.end());
        pre.push_back(e.HT);
        std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> ras;
        for (uint32_t c : cols) {
            for (uint32_t cy = 0; cy < GY; ++cy) {
                ras.emplace_back(CoreCoord(c, cy), prep_compute(rs.compute_args(p, c, cy), pre.size(), c, cy));
            }
        }
        size_t shared = 0;
        const auto& first = ras[0].second;
        while (shared < first.size() &&
               std::all_of(
                   ras.begin(),
                   ras.end(),
                   [&](const auto& r) { return r.second.size() > shared && r.second[shared] == first[shared]; })) {
            ++shared;
        }
        std::vector<uint32_t> common_c = pre;
        common_c.insert(common_c.end(), first.begin(), first.begin() + shared);
        auto compute = kernel("compute/se_compute.cpp", CoreRangeSet(rr), comp);
        compute.compile_time_args = {
            MT, e.KB, e.NPG, e.NCH, e.POMAX, e.KBD, e.NSD, e.NPD, e.WSLOT, so[1], so[0], GX, u32(common_c.size())};
        define(compute, "SE_GLU", iv(glu_code));
        for (const auto& [k, v] : glu_vals) {
            define(compute, k, float_literal(v));
        }
        compute.defines.insert(compute.defines.end(), nd.begin(), nd.end());
        for (const auto& [c, ra_] : ras) {
            compute.runtime_args.emplace_back(
                c,
                ra_.size() > shared ? std::vector<uint32_t>(ra_.begin() + shared, ra_.end())
                                    : std::vector<uint32_t>{0});
        }
        compute.common_runtime_args = common_c;
        push(ks, compute);
    }
    return ks;
}

// ---- x_gather3.XGather3 ----

Args ChipProgram::rs_block(XDir d, uint32_t v) const {
    const RsDir rd = rs_dir(d);
    const auto q = rs.down(p, rd);
    const auto units = q ? rs.port_units(p, rd) : std::vector<RsPlan::PortUnit>{};
    const auto cores = units.empty() ? std::vector<std::pair<uint32_t, uint32_t>>{} : rs.port_cores(v);
    Args out = {u32(units.size())};
    for (const auto& u : units) {
        const uint32_t pm0 = rs.compact ? rs.prefix_mask(*q, 0, rd, u.slot0) : 0;
        const uint32_t pm1 = rs.compact ? rs.prefix_mask(*q, 1, rd, u.slot1) : 0;
        out.push_back(
            u.entry | u.block << 8 | u.mask << 16 | pm0 << 20 | pm1 << 24 | static_cast<uint32_t>(u.pieces) << 28);
        out.push_back(u.slot0 | u.slot1 << 16);
    }
    out.push_back(u32(cores.size()));
    for (const auto& [cx, cy] : cores) {
        const auto c = h.evirt(cx, cy);
        const uint32_t k = rs.k(cx, cy);
        uint32_t pn = 0;
        if (rs.pm) {
            for (uint32_t w_ : rs.pass_widths(cx)) {
                pn += w_;
            }
        }
        TT_FATAL(k < 128 && pn < 32, "fused_head: a core word's k {} < 128, pn {} < 32", k, pn);
        out.push_back(u32(c.x | c.y << 8 | rs.half(k) << 16 | k << 20 | pn << 27));
        out.push_back(rs.counts_word(cx, cy));
    }
    const int i = static_cast<int>(d);
    auto lands = [&](uint32_t qq, uint32_t cx, uint32_t cy, RsDir dd) {
        const auto arr_ = rs.arrivals(qq, cx, cy);
        for (const auto& a : arr_[static_cast<int>(dd)]) {
            if (a.threshold) {
                return true;
            }
        }
        return false;
    };
    const uint32_t rd_ = e.rtr_zones ? 1 : 0;
    const RsDir od = rd == RsDir::Fwd ? RsDir::Bwd : RsDir::Fwd;
    uint32_t zneed = 0, zexp = 0;
    if (e.NCH > 1 || rd_) {
        for (const auto& [cx, cy] : rs.port_cores(v)) {
            zneed += lands(p, cx, cy, od) ? 1 : 0;
        }
        if (q) {
            bool any = false;
            for (const auto& [cx, cy] : rs.port_cores(v)) {
                any = any || lands(*q, cx, cy, rd);
            }
            zexp = any ? e.NCH - 1 + rd_ : 0;
        }
    }
    const auto& ents = rs.entries(p);
    out.insert(out.end(), ents.begin(), ents.end());
    for (const Arg& x : std::initializer_list<Arg>{
             A("finb"),
             mbox,
             i == 0 ? A("arrf") : A("arrb"),
             arr[i],
             rs.compact ? 0u : rs.P,
             arr.size() > 3 ? arr[3] : Arg(0u),
             zfree + 16,
             zfree,
             zneed,
             zexp,
             e.rtr_overlay ? zfree + 32 : 0u,
             e.rtr_overlay ? (e.GX * e.GY | rd_ << 16 | static_cast<uint32_t>(e.rtr_latz) << 17) : 0u}) {
        out.push_back(x);
    }
    return out;
}

Args ChipProgram::la_block(XDir d, uint32_t v) const {
    if (!e.LT) {
        return Args{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    }
    auto ladir = [&](uint32_t a, uint32_t b) {
        return rs.kind == RsKind::Line ? (b > a ? XDir::Fwd : XDir::Bwd) : h.ring_dir(a, b);
    };
    std::vector<uint32_t> dst;
    for (uint32_t b = 0; b < h.G; ++b) {
        if (b != p && ladir(p, b) == d) {
            dst.push_back(b);
        }
    }
    TT_FATAL(dst.size() <= 3 && h.group + 1 + dst.size() <= 12, "fused_head: packet headers: 12 a RISC");
    Args out = {u32(dst.size())};
    for (uint32_t b : dst) {
        const auto n = h.node(h.coord_of_pos(coord, b));
        out.push_back(b | (p < b ? p : p - 1) << 8 | u32(n.chip_id) << 16);
        out.push_back(u32(*n.mesh_id));
    }
    std::vector<uint32_t> cw;
    uint32_t ncores = 0;
    for (uint32_t cy = 0; cy < e.GY; ++cy) {
        for (uint32_t cx = 0; cx < e.GX; ++cx) {
            if ((cy * e.GX + cx) % h.L != v) {
                continue;
            }
            const auto c = h.evirt(cx, cy);
            uint32_t rows = 0;
            for (uint32_t b = 0; b < h.G; ++b) {
                const auto [r0, nr] = e.la_rows(cy, b);
                rows |= (r0 | nr << 2) << (4 * b);
            }
            cw.push_back(u32(c.x | c.y << 8 | (cy * e.GX + cx) << 16));
            cw.push_back(rows);
            ++ncores;
        }
    }
    out.push_back(ncores);
    out.insert(out.end(), cw.begin(), cw.end());
    const Arg fence = mbox + h.HS * e.GX * e.GY;
    std::vector<uint32_t> sends;
    for (uint32_t q = 0; q < h.G; ++q) {
        if (q == p || ladir(p, q) != d) {
            continue;
        }
        const Coord qc = h.coord_of_pos(coord, q);
        const auto pv = h.virt(h.chip(qc).port(ladir(q, p), v));
        const auto nq = h.node(qc);
        sends.push_back(u32(nq.chip_id) | u32(*nq.mesh_id) << 16);
        sends.push_back(u32(pv.x | pv.y << 8));
    }
    for (const Arg& x : std::initializer_list<Arg>{
             A("lat"), fence + 128, A("latz"), arr[2], e.LWMAX, fence, u32(dst.size()), u32(sends.size() / 2)}) {
        out.push_back(x);
    }
    out.insert(out.end(), sends.begin(), sends.end());
    return out;
}

// router_stream.RouterStream.port_block: the port sender's logits args
Args ChipProgram::port_block(XDir d, uint32_t v) const {
    if (!h.chip(coord).rg.peer(d)) {
        return Args(11, 0u);
    }
    const auto bs = h.dests(p, d);
    const auto js = h.port_layout(v);
    Args out = {u32(bs.size())};
    for (uint32_t b : bs) {
        const auto n = h.node(h.coord_of_pos(coord, b));
        out.push_back(u32(n.chip_id) | (p < b ? p : p - 1) << 16);
        out.push_back(u32(*n.mesh_id));
    }
    out.push_back(u32(js.size()));
    for (uint32_t j : js) {
        const auto c = h.virt(h.rcores[j]);
        out.push_back(u32(c.x | c.y << 8));
    }
    const uint32_t rpl = std::max<uint32_t>(1, h.payload / 4096);
    TT_FATAL(bs.size() * js.size() <= h.port_words, "fused_head: the port's sub-block words: {} a port", h.port_words);
    for (const Arg& x : std::initializer_list<Arg>{
             base,
             mbox + 4 * h.lw_off,
             RA("arr"),
             rt_arr0,
             h.ET,
             rpl,
             h.ET,
             std::min<uint32_t>(h.NS, u32(js.size())),
             sfree + 64 * static_cast<uint32_t>(d)}) {
        out.push_back(x);
    }
    return out;
}

Args ChipProgram::lg_block(XDir d, uint32_t v) const {
    const Arg fence = zfree + 48;
    if (!h.chip(coord).rg.peer(d)) {
        return Args{0u, 0u, 0u};
    }
    const auto js = h.port_layout(v);
    const uint32_t need = js.empty() ? 0 : u32(h.dests(p, d).size());
    std::vector<uint32_t> sends;
    for (uint32_t q = 0; q < h.G; ++q) {
        if (q == p || h.rdir(p, q) != d || js.empty()) {
            continue;
        }
        const Coord qc = h.coord_of_pos(coord, q);
        const XDir dq = h.rdir(q, p);
        if (!h.chip(qc).rg.peer(dq)) {
            continue;
        }
        const auto pv = h.virt(h.chip(qc).port(dq, v));
        const auto nq = h.node(qc);
        sends.push_back(u32(nq.chip_id) | u32(*nq.mesh_id) << 16);
        sends.push_back(u32(pv.x | pv.y << 8));
    }
    Args out = {(need || !sends.empty()) ? fence : 0u, need, u32(sends.size() / 2)};
    out.insert(out.end(), sends.begin(), sends.end());
    return out;
}

std::vector<KernelDescriptor> ChipProgram::gather(std::vector<CBDescriptor>& cbs) {
    const auto& ch = h.chip(coord);
    const auto& rg = ch.rg;
    const bool xdram = h.cfg.lay.xdram;
    const Arg xown = A("xown");
    const auto T = h.tables(coord);
    std::vector<CoreCoord> pcores;
    for (const auto& [d, l, c] : ch.ports) {
        pcores.push_back(c);
    }
    const auto pset = cells(pcores);
    // (in the arena, as every head CB: no static CB under the arena)
    cbs.push_back(arena_cb(kXgCb, h.xg_off, u32(h.xg_bytes), tt::DataFormat::Float16_b, 2048, pset));
    const std::vector<uint32_t> ct = {
        e.KB, e.MT, h.sps, h.rp, e.CR, e.NCH, h.L, h.group, kIncEvery, e.GX, e.GY, e.S0, e.S1, h.G};
    const uint32_t npass = rs.pm ? rs.npass : 1;
    const std::string la_chl = uv(e.NCH > 1 ? e.LATP * 2048 : 0);
    auto reader = kernel("dataflow/xg_reader.cpp", pset, dm(DataMovementProcessor::RISCV_0, NOC::RISCV_1_default));
    reader.compile_time_args = ct;
    append_accessor(reader.compile_time_args, bufs[SRC_X]);
    define(reader, "XG_HS", uv(h.HS));
    define(reader, "LA_CHL", la_chl);
    define(reader, "RS_NPASS", uv(npass));
    auto sender = kernel("dataflow/xg_sender.cpp", pset, dm(DataMovementProcessor::RISCV_1, NOC::RISCV_0_default));
    sender.compile_time_args = ct;
    define(sender, "XG_HS", uv(h.HS));
    define(sender, "LS_FIRST", iv(h.logits_first()));
    define(sender, "LA_CHL", la_chl);
    define(sender, "RS_NPASS", uv(npass));
    routing_defines(sender);
    routing_accessors(sender);
    for (const auto& [d, v, core] : ch.ports) {
        auto pl = h.port_plan(coord, d, v);
        const bool fence = !pl.mine.empty();
        if (xdram) {
            pl.mine.clear();
            pl.up_ipc = 0;
            pl.expect_in = 0;
        }
        std::vector<uint32_t> ent_r, ent_s;
        for (const auto& [q, part, ub, npk] : pl.mine) {
            ent_r.insert(ent_r.end(), {q, part, ub, npk});
            ent_s.insert(ent_s.end(), {q, part, npk});
        }
        const Args rsb = rs_block(d, v);
        Args rd = {S(SRC_X)};
        add(rd, Args{xown, v, arrival, u32(pl.mine.size()), pl.up_ipc});
        add(rd, ent_r);
        add(rd, T);
        add(rd, Args{xloaded, e.GX * e.GY});
        add(rd, rsb);
        add(rd, la_block(d, v));
        set_rt(reader, core, rd);

        Args blocks = rsb;
        add(blocks, port_block(d, v));
        add(blocks, la_block(d, v));
        add(blocks, lg_block(d, v));
        add(blocks, send_block(SendKey::x(d, v)));
        add(blocks, fwd_block(SendKey::x(d, v)));
        Args args = {xown, v, arrival, pl.expect_in, ready, u32(fence)};
        const auto& peer = rg.peer(d);
        if (peer && !rg.links[static_cast<int>(d)].empty()) {
            const auto& pch = h.chip(*peer);
            const auto rc = h.virt(pch.port(opp(d), v)), pcv = h.virt(pch.port(d, v)), wc = h.virt(h.watcher);
            const auto pn = h.node(*peer);
            add(args,
                Args{
                    u32(!pch.rg.sched[static_cast<int>(opp(d))].empty()),
                    u32(rc.x),
                    u32(rc.y),
                    u32(pcv.x),
                    u32(pcv.y),
                    u32(*pn.mesh_id),
                    u32(pn.chip_id),
                    u32(wc.x),
                    u32(wc.y),
                    watch.at({d, v}),
                    u32(pl.mine.size())});
            add(args, ent_s);
            add(args, h.tables(*peer));
            add(args, blocks);
            add(args, fabric(*peer, rg.links[static_cast<int>(d)][v], core));
        } else {
            add(args, std::vector<uint32_t>(11, 0));
            add(args, T);
            add(args, blocks);
        }
        set_rt(sender, core, args);
    }
    // the watcher
    std::vector<CoreCoord> ecores;
    for (uint32_t c = 0; c < e.GX; ++c) {
        for (uint32_t cy = 0; cy < e.GY; ++cy) {
            ecores.push_back(h.evirt(c, cy));
        }
    }
    const auto [shards, carry] = h.shard_order(coord);
    Args wargs;
    std::vector<std::pair<Arg, uint32_t>> finals;
    if (!xdram) {
        for (size_t k = 1; k < shards.size(); ++k) {
            const auto& cs = carry.at(shards[k]);
            wargs.push_back(u32(cs.size()));
            for (const auto& [dd, vv, j, n] : cs) {
                add(wargs, Args{watch.at({dd, vv}), j, n});
            }
        }
        for (XDir d : {XDir::Fwd, XDir::Bwd}) {
            const auto& up = d == XDir::Fwd ? rg.prev : rg.next;
            if (up && !h.chip(*up).rg.sched[static_cast<int>(d)].empty()) {
                for (uint32_t v = 0; v < h.L; ++v) {
                    finals.emplace_back(
                        watch.at({d, v}), e.NCH * u32(h.chip(*up).rg.sched[static_cast<int>(d)].size()));
                }
            }
        }
    }
    Args wl = {e.NCH, kXready, xdram ? 0u : u32(shards.size() - 1), u32(ecores.size()), u32(finals.size())};
    for (const auto& [k, f] : finals) {
        add(wl, Args{k, f});
    }
    for (const auto& c : ecores) {
        add(wl, Args{u32(c.x), u32(c.y)});
    }
    add(wl, wargs);
    auto watcher =
        kernel("dataflow/xg_watcher.cpp", cells({h.watcher}), dm(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default));
    define(watcher, "XG_HS", uv(h.HS));
    std::vector<SendKey> ykeys;
    for (const auto& [k, _] : sv) {
        if (k.y) {
            ykeys.push_back(k);
        }
    }
    if (ykeys.size() >= 2) {  // (the second Y direction's sender, after the watch: RoutingStream.watcher_tail)
        routing_defines(watcher);
        define(watcher, "W_RA0", iv(wl.size()));
        const auto tail = y_args(ykeys[1], h.watcher);
        add(wl, tail);
    }
    set_rt(watcher, h.watcher, wl);
    routing_accessors(watcher);
    std::vector<KernelDescriptor> ks;
    push(ks, reader);
    push(ks, sender);
    push(ks, watcher);
    return ks;
}

// ---- routing_stream.RoutingStream ----

void ChipProgram::routing_defines(KernelDescriptor& k) const {
    for (const auto& [n, v] : std::vector<std::pair<std::string, int64_t>>{
             {"RA_NC_", int64_t(h.rcores.size())},
             {"RA_NP_", h.RB},
             {"RA_N_", h.N},
             {"RA_M_", h.M},
             {"RA_EPC_", h.EPC},
             {"RA_ME_", me},
             {"RA_HBP_", h.HBP},
             {"RA_BB_", h.BB},
             {"RA_BBP_", h.BBP},
             {"RA_MAXP_", h.MAXP},
             {"RA_NF_", h.NF}}) {
        define(k, n, iv(v));
    }
}

// router_stream._ra_src: the planner as ra::plan (RAP_* = RoutingStream.kernel_subs)
void ChipProgram::planner_defines(KernelDescriptor& k, uint32_t ct0) const {
    define(k, "RA_PLAN");
    routing_defines(k);
    for (const auto& [n, v] : std::vector<std::pair<std::string, int64_t>>{
             {"RAP_S", h.S},
             {"RAP_K", h.K},
             {"RAP_EPC", h.EPC},
             {"RAP_R", h.R},
             {"RAP_C", h.C},
             {"RAP_ME", me},
             {"RAP_A", h.A},
             {"RAP_L", h.Lr},
             {"RAP_P", h.luts.num_ports.at(ChipCoord{coord.first, coord.second})},
             {"RAP_TPP", 32},
             {"RAP_M", h.M},
             {"RAP_HBP", h.HBP},
             {"RAP_BB", h.BB},
             {"RAP_BBP", h.BBP},
             {"RAP_PCAP", h.PCAP},
             {"RAP_NPORT", int64_t(sv.size())},
             {"RAP_ROWS_CAP", h.rows_cap},
             {"RAP_ENT_CAP", h.inst_cap},
             {"RAP_WX", int64_t(h.ring)},
             {"RAP_WY", int64_t(h.wrap_y)},
             {"RAP_CT0", ct0}}) {
        define(k, n, iv(v));
    }
}

std::vector<uint32_t> ChipProgram::next_hop(uint32_t s, uint32_t d, uint32_t i) const {
    const auto steps = h.path(s, d);
    if (i + 1 < steps.size()) {
        const auto [ci, key] = steps[i + 1];
        const Coord cc{ci / h.C, ci % h.C};
        const auto core = h.virt(h.core_of(cc, key));
        const uint32_t slot = h.fwd_at.at({s, d, i + 1});
        const uint32_t ki = u32(std::find(h.fkeys.begin(), h.fkeys.end(), key) - h.fkeys.begin());
        return {u32(h.node(cc).chip_id), 1, u32(core.x), u32(core.y), slot | (ki * h.NF + slot) << 8};
    }
    return {u32(h.node_idx(d).chip_id), 0, 0, 0, 0};
}

Args ChipProgram::fwd_block(const SendKey& k) const {
    auto it = h.fwd.find({me, k});
    if (it == h.fwd.end() || it->second.empty()) {
        return {0u};
    }
    const auto& lst = it->second;
    const auto p0 = h.virt(h.rcores[0]);
    const auto nd = h.node(coord);
    TT_FATAL(lst.size() <= h.NF, "fused_head: routing: a port forwards {} packets > its {} slots", lst.size(), h.NF);
    const uint32_t ki = u32(std::find(h.fkeys.begin(), h.fkeys.end(), k) - h.fkeys.begin());
    Args out = {u32(lst.size()), base + u32(h.fbase_off), mbox + h.HS * 2 * e.GX * e.GY + 128, u32(h.PCAP)};
    add(out, S(SRC_SLOTS));
    add(out, Args{ra.at("c0"), ra.at("c1"), u32(p0.x), u32(p0.y), u32(*nd.mesh_id)});
    add(out, S(SRC_FSLOTS));
    add(out, ki * h.NF);
    for (const auto& [s_, d, i] : lst) {
        add(out, next_hop(s_, d, i));
    }
    return out;
}

Args ChipProgram::send_block(const SendKey& k) const {
    auto it = sv.find(k);
    if (it == sv.end()) {
        return {0u};
    }
    const auto& dests = it->second;
    const uint32_t self_ = dests.back() == me ? 1 : 0;
    const uint32_t nr = u32(dests.size()) - self_;
    const auto p0 = h.virt(h.rcores[0]);
    std::vector<SendKey> ykeys;
    for (const auto& [kk, _] : sv) {
        if (kk.y) {
            ykeys.push_back(kk);
        }
    }
    const bool second_y = k.y && k != ykeys[0];
    Args out = {u32(dests.size()), nr, self_, base + u32(h.buf_off(coord, k)), u32(h.PCAP)};
    add(out, S(SRC_SLOTS));
    add(out,
        Args{
            ra.at("c0"),
            ra.at("c1"),
            u32(p0.x),
            u32(p0.y),
            ra.at(second_y ? "calls_s2" : "calls_s"),
            ra.at(second_y ? "ready2" : "ready"),
            u32(*h.node(coord).mesh_id)});
    add(out, S(SRC_FSLOTS));
    add(out, mbox + h.HS * 2 * e.GX * e.GY + 128);
    for (uint32_t d : dests) {
        add(out, d != me ? next_hop(me, d, 0) : std::vector<uint32_t>(5, 0));
    }
    return out;
}

Args ChipProgram::y_args(const SendKey& k, const CoreCoord& core) {
    const auto& dests = sv.at(k);
    const uint32_t nr = u32(dests.size()) - (dests.back() == me ? 1 : 0);
    Args args = send_block(k);
    add(args, u32(nr > 0));
    if (nr) {
        const int64_t dy = k.a == 2 ? 1 : -1;
        const Coord peer{u32(pmod(int64_t(coord.first) + dy, h.R)), coord.second};
        add(args, word("yhello"));
        add(args, Args{u32(h.node(peer).chip_id), u32(*h.node(peer).mesh_id)});
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*h.mesh, h.node(coord), h.node(peer));
        TT_FATAL(!links.empty(), "fused_head: no link toward the Y neighbour");
        add(args, fabric(peer, links[0], core));
        // (after the connection's args: the hellos to wait for - ra_y_send's, the first Y direction's; the watcher's
        // second direction sends none and ignores it)
        add(args, h.y_hellos_in(coord));
    }
    return args;
}

std::vector<KernelDescriptor> ChipProgram::y_kernels() {
    std::vector<KernelDescriptor> ks;
    for (const auto& [k, _] : sv) {
        if (!k.y) {
            continue;
        }
        const CoreCoord core = h.watcher;
        auto y =
            kernel("dataflow/ra_y_send.cpp", cells({core}), dm(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default));
        routing_defines(y);
        routing_accessors(y);
        set_rt(y, core, y_args(k, core));
        push(ks, y);
        break;  // (the first Y direction; the second rides the watcher's NCRISC)
    }
    return ks;
}

std::function<Args(uint32_t)> ChipProgram::planner_args() const {
    std::map<uint32_t, std::tuple<CoreCoord, uint32_t, int64_t>> where;
    for (const auto& [key, ds] : sv) {
        const CoreCoord core = h.core_of(coord, key);
        const int64_t b = h.buf_off(coord, key);
        for (uint32_t j = 0; j < ds.size(); ++j) {
            where[ds[j]] = {core, j, b};
        }
    }
    Args head = {S(SRC_ROWS), S(SRC_SLOTS), ra.at("c0"), ra.at("c1"), 0u};
    for (HeadSrc s : {SRC_COUNTS, SRC_REGIONS, SRC_TOKEN_INDEX, SRC_TABLE, SRC_PLAN}) {
        add(head, S(s));
    }
    const auto& lut = h.luts.lut.at(ChipCoord{coord.first, coord.second});
    for (uint8_t v : lut) {
        add(head, uint32_t(v));
    }
    const uint32_t nc = u32(h.rcores.size());
    std::vector<uint32_t> pxy;
    for (uint32_t q = 0; q < h.M; ++q) {
        const auto v = h.virt(h.rcores[q % nc]);
        pxy.push_back(u32(v.x));
        pxy.push_back(u32(v.y));
    }
    Args inst;
    for (int64_t pb : h.pbases_off) {
        inst.push_back(base + u32(pb));
    }
    for (uint32_t q = 0; q < h.M; ++q) {
        inst.push_back(q < nc ? ra.at("ex") : ra.at("ex2"));
    }
    for (uint32_t q = 0; q < h.M; ++q) {
        inst.push_back(q < nc ? ra.at("go") : ra.at("go2"));
    }
    Args dst;
    for (uint32_t d = 0; d < h.N; ++d) {
        const auto& [core, j, b] = where.at(d);
        const auto v = h.virt(core);
        add(dst, Args{u32(v.x), u32(v.y), base + u32(b) + j * u32(h.PCAP)});
    }
    std::vector<SendKey> ykeys;
    for (const auto& [kk, _] : sv) {
        if (kk.y) {
            ykeys.push_back(kk);
        }
    }
    Args port_args;
    for (const auto& [k, _] : sv) {
        const auto v = h.virt(h.core_of(coord, k));
        const bool second_y = k.y && k != ykeys[0];
        add(port_args, Args{u32(v.x), u32(v.y), ra.at(second_y ? "ready2" : "ready")});
    }
    const Args tail = {ra.at("ready")};
    auto ra_ = ra;
    return [=](uint32_t q) {
        Args a = head;
        a[4] = q < nc ? ra_.at("calls_p") : ra_.at("calls_p2");
        add(a, Args{q, 0, 0});
        add(a, pxy);
        add(a, inst);
        add(a, dst);
        add(a, port_args);
        add(a, tail);
        add(a, Args{q < nc ? ra_.at("go") : ra_.at("go2"), ra_.at("parts"), q < nc ? ra_.at("ex") : ra_.at("ex2")});
        return a;
    };
}

Args ChipProgram::ra_args(uint32_t j) const {
    std::vector<uint32_t> pxy;
    for (const auto& c : h.rcores) {
        const auto v = h.virt(c);
        pxy.push_back(u32(v.x));
        pxy.push_back(u32(v.y));
    }
    if (j >= h.M) {
        Args a = {ra.at("rows"), ra.at("rows2"), 0u};
        add(a, pxy);
        return a;
    }
    Args a = {ra.at("rows"), ra.at("rows2"), base + u32(h.pbases_off[j])};
    add(a, pxy);
    add(a, planner_args()(j));
    return a;
}

Args ChipProgram::ra_args2(uint32_t j) const {
    const uint32_t q = j + u32(h.rcores.size());
    if (q >= h.M) {
        return {ra.at("rows2"), 0u};
    }
    Args a = {ra.at("rows2"), base + u32(h.pbases_off[q])};
    add(a, planner_args()(q));
    return a;
}

// ---- router_stream.RouterStream.pieces ----

std::vector<KernelDescriptor> ChipProgram::router(std::vector<CBDescriptor>& cbs) {
    const uint32_t G = h.G, ET = h.ET, RB = h.RB, K = h.K;
    const auto crs = cells(h.rcores);
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;
    auto cb = [&](uint8_t i, const std::string& key, uint32_t size, tt::DataFormat dt, uint32_t page) {
        cbs.push_back(arena_cb(i, region_of_router(key), size, dt, page, crs));
    };
    cb(RT_HOLD, "hold", G * ET * 4096, fp32, 4096);
    cb(ARRF, "arr", (G - 1) * ET * 4096, fp32, 4096);
    cb(BIAS, "bias", h.ET * 4096, fp32, 4096);
    cb(SC, "sc", ET * 4096, fp32, 4096);
    cb(IDX, "idx", ET * 2048, tt::DataFormat::UInt16, 2048);
    cb(BEST, "best", 4096, fp32, 4096);
    cb(PSYNC, "psync", 16 * 2, bf16, 16);
    cb(ARRB, "arrb", 16 * std::max<uint32_t>(1, G - 1), tt::DataFormat::UInt32, 16);
    cb(RDY, "rdy", 16 * 2, bf16, 16);
    float scale = h.cfg.route_scale;
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    const Args tail = {RA("words"), RA("best"), RA("out"), scale_bits};
    const Arg sv_ = RA("hold") + p * ET * 4096;
    const auto order = h.arrival_order(p);
    std::vector<std::pair<CoreCoord, Args>> n_rt, c_rt, b_rt;
    for (uint32_t j = 0; j < h.rcores.size(); ++j) {
        const CoreCoord core = h.rcores[j];
        Args n = {S(SRC_RB), S(SRC_RIX)};
        for (uint32_t bb = 0; bb < G; ++bb) {
            add(n, bb * RB + j % RB);
        }
        add(n, Args{G - 1, rt_arr0});
        add(n, order);
        add(n, tail);
        add(n, sv_);
        add(n, ra_args2(j));
        n_rt.emplace_back(core, n);
        Args c = {1u, 2 | p << 8};  // (one op: the fin of my owner block)
        for (uint32_t bb = 0; bb < G; ++bb) {
            add(c, bb == p ? G - 1 : 0u);
        }
        add(c, tail);
        add(c, Args{sv_, RA("route"), 32 * (j % RB)});
        c_rt.emplace_back(core, c);
        const uint32_t jr = j % RB;
        Args b = {S(SRC_ROWS)};
        add(b, jr);
        add(b, Args{scale_bits, RA("out"), RA("words"), RA("best"), RA("route")});
        b.push_back(S(SRC_ROUTE));
        add(b, Args{32 * jr, sv_});
        add(b, ra_args(j));
        b_rt.emplace_back(core, b);
    }
    for (const auto& lst : {n_rt, c_rt, b_rt}) {
        for (const auto& [core, a] : lst) {
            TT_FATAL(a.size() <= 341, "fused_head: router core runtime args {} > 341", a.size());
        }
    }
    const std::vector<uint32_t> ct = {h.KT, ET, G, K};
    std::vector<uint32_t> ra_ct;
    append_accessor(ra_ct, bufs[SRC_ROWS]);
    append_accessor(ra_ct, bufs[SRC_SLOTS]);
    std::vector<uint32_t> n_ct = ct;
    for (Buffer* bb : {bufs[SRC_RB], bufs[SRC_RIX]}) {
        append_accessor(n_ct, bb);
    }
    const uint32_t n_ra0 = u32(n_ct.size());
    n_ct.insert(n_ct.end(), ra_ct.begin(), ra_ct.end());
    std::vector<uint32_t> b_ct = ct;
    append_accessor(b_ct, bufs[SRC_ROWS]);
    const uint32_t b_ra0 = u32(b_ct.size());
    b_ct.insert(b_ct.end(), ra_ct.begin(), ra_ct.end());
    const uint32_t rt_ct0 = u32(b_ct.size());
    ComputeConfigDescriptor comp{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };
    comp.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint8_t i : {SC, BEST, RT_HOLD, ARRF, BIAS}) {
        comp.unpack_to_dest_mode[i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    const Args common = {base, rt_arr1};
    // a padded chunk (router_addr.hpp RT_VALID): the planners and the route words' writer read actual_start /
    // actual_end (common [2], [3]) and bound this chip's tokens by its SP row's real rows (main's rotated layout: [4]
    // the SP row r, [5] SP rows, [6] tokens a row) less the row's tokens before mine ([7] p S, my owned block p), [8]
    // S; absent, the programs are the unpadded ones
    const bool padded = bufs[SRC_START] != nullptr;
    Args common_v = common;
    if (padded) {
        add(common_v,
            Args{S(SRC_START), S(SRC_END), Arg(coord.first), Arg(h.R), Arg(h.S * h.C), Arg(p * h.S), Arg(h.S)});
    }
    auto eng_defs = [&](KernelDescriptor& k) {
        define(k, "RT_MSYNC_OFF", uv(region_of_router("msync")));
        define(k, "RT_FOLD", uv(h.cfg.shape.fold));
        if (h.cfg.ordered) {
            define(k, "RT_ARR_ORDERED");
        }
    };
    const SpanTables spans = span_subs(h.R, h.C, h.ring, h.wrap_y);
    auto join = [](const auto& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) {
            s += (i ? "," : "") + std::to_string(uint32_t(v[i]));
        }
        return s;
    };
    std::vector<KernelDescriptor> ks;
    {
        const auto& hcrs = crs;
        auto mine = [&](const std::vector<std::pair<CoreCoord, Args>>& lst, KernelDescriptor& k) {
            for (const auto& [core, a] : lst) {
                set_rt(k, core, a);
            }
        };
        auto nk = kernel("dataflow/router_ncrisc.cpp", hcrs, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        eng_defs(nk);
        planner_defines(nk, n_ra0);
        nk.compile_time_args = n_ct;
        mine(n_rt, nk);
        if (padded) {
            define(nk, "RT_VALID");
        }
        set_common(nk, common_v);
        routing_accessors(nk);
        auto ck = kernel("compute/router_compute.cpp", hcrs, comp);
        eng_defs(ck);
        if (h.cfg.sqrtsoftplus) {
            define(ck, "RT_SQRTSOFTPLUS");
            define(ck, "INP_FLOAT32", "1");
        }
        define(ck, "RT_ROUTE");
        define(ck, "RW_R", iv(h.R));
        define(ck, "RW_C", iv(h.C));
        define(ck, "RW_ME", iv(me));
        define(ck, "RW_K", iv(K));
        define(ck, "RW_RW", iv(route_words(h.N)));
        define(ck, "RW_E", iv(h.ET * 32 / h.cfg.shape.fold));
        define(ck, "RW_WRAPX", spans.wrap_x ? "true" : "false");
        define(ck, "RW_WRAPY", spans.wrap_y ? "true" : "false");
        define(ck, "RW_XRING_INIT", join(spans.xring));
        define(ck, "RW_YRING_INIT", join(spans.yring));
        std::vector<uint32_t> chips_of;
        for (uint32_t ee = 0; ee < h.ET * 32 / h.cfg.shape.fold; ++ee) {
            chips_of.push_back(ee / h.EPC);
        }
        define(ck, "RW_CHIPS_INIT", join(chips_of));
        ck.compile_time_args = ct;
        mine(c_rt, ck);
        set_common(ck, common);
        auto bk = kernel("dataflow/router_brisc.cpp", hcrs, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        define(bk, "RT_FOLD", uv(h.cfg.shape.fold));
        bk.compile_time_args = b_ct;
        define(bk, "RT_ROUTE");
        define(bk, "RT_CT0", iv(rt_ct0));
        define(bk, "RT_PAGE", iv(bufs[SRC_ROUTE]->aligned_page_size()));
        define(bk, "RT_RW", iv(route_words(h.N)));
        append_accessor(bk.compile_time_args, bufs[SRC_ROUTE]);
        planner_defines(bk, b_ra0);
        mine(b_rt, bk);
        if (padded) {
            define(bk, "RT_VALID");
        }
        set_common(bk, common_v);
        routing_accessors(bk);
        push(ks, nk);
        push(ks, ck);
        push(ks, bk);
    }
    return ks;
}

ProgramDescriptor ChipProgram::build(HeadPatches* patches) {
    namespace kh = ttnn::kernel_lib::host;
    // the engine's semaphores (its kernels' fixed ids), also on the watcher
    const CoreRangeSet semc = h.engine_cores().merge(cells({h.watcher}));
    for (uint32_t i = 0; i < kNsemOwn; ++i) {
        desc.semaphores.push_back(SemaphoreDescriptor{.id = i, .core_ranges = semc, .initial_value = 0});
    }
    std::vector<CBDescriptor> c_x, c_r, c_e;
    take_pushed();
    auto kx = gather(c_x);
    auto px = take_pushed();
    auto kr = router(c_r);
    auto pr = take_pushed();
    auto ky = y_kernels();
    auto py = take_pushed();
    auto ke = engine(c_e);
    auto pe = take_pushed();
    TT_FATAL(
        px.size() == kx.size() && pr.size() == kr.size() && py.size() == ky.size() && pe.size() == ke.size() &&
            kpatches.empty(),
        "fused_head: a kernel's address args not carried to the assembly");
    for (auto* v : {&c_x, &c_r, &c_e}) {
        desc.cbs.insert(desc.cbs.end(), v->begin(), v->end());
    }
    // the kernel_lib families: the engine's x slots 0 / 1 and h on its BRISC (a receiver group a row)
    const auto rect = h.engine_cores();
    auto cols = [&](const std::vector<uint32_t>& cs) {
        std::vector<CoreRange> rr;
        for (uint32_t c : cs) {
            rr.emplace_back(CoreCoord(c, 0), CoreCoord(c, e.GY - 1));
        }
        return CoreRangeSet(rr);
    };
    auto owners = [&](const std::vector<uint32_t>& fam) -> kh::McastSenderConfig {
        if (fam.size() > 1) {
            return kh::McastRotatingSenderConfig{.sender_cores = cols(fam)};
        }
        return kh::McastFixedSenderConfig{.sender_index = fam[0]};
    };
    uint32_t next_id = kNsemOwn;
    for (const auto& [name, cfg] : std::vector<std::pair<std::string, kh::McastSenderConfig>>{
             {"x0", owners(e.fam0)}, {"x1", owners(e.fam1)}, {"h", kh::McastRotatingSenderConfig{}}}) {
        kh::Mcast mc(*h.mesh, kh::McastConfig{.noc = NOC::NOC_0, .handshake = true}, rect, e.GX, cfg);
        const std::array kernels{std::ref(ke[0])};
        mc.attach(desc, name, kernels, next_id);
        next_id = mc.next_semaphore_id();
    }
    for (auto [v, p] : {std::pair{&kx, &px}, std::pair{&kr, &pr}, std::pair{&ky, &py}, std::pair{&ke, &pe}}) {
        for (size_t i = 0; i < v->size(); ++i) {
            if (patches) {
                for (auto a : (*p)[i]) {
                    a.kernel = u32(desc.kernels.size());
                    patches->args.push_back(a);
                }
            }
            desc.kernels.push_back(std::move((*v)[i]));
        }
    }
    if (patches) {
        std::stable_sort(
            patches->args.begin(), patches->args.end(), [](const auto& a, const auto& b) { return a.src < b.src; });
        for (uint32_t i = 0; i < patches->args.size();) {
            uint32_t j = i + 1;
            while (j < patches->args.size() && patches->args[j].src == patches->args[i].src) {
                ++j;
            }
            patches->groups.emplace_back(i, j);
            i = j;
        }
        for (uint32_t i = 0; i < desc.cbs.size(); ++i) {
            if (desc.cbs[i].buffer != nullptr) {
                patches->arena_cbs.push_back({i, desc.cbs[i].address_offset});
            }
        }
    }
    return std::move(desc);
}

HeadTensorShapes head_tensor_shapes(const HeadModel& h) {
    const auto& sh = h.cfg.shape;
    HeadTensorShapes s{};
    const bool width = sh.rs_dim() == -1;
    s.y_rows = width ? sh.rows : sh.rows / h.G;
    s.y_cols = width ? sh.H / h.G : sh.H;
    s.rows_rows = h.RB;
    s.rows_cols = 32 * 2 * h.K;
    s.xbuf_rows = h.N * sh.rows / h.G;
    s.xbuf_cols = sh.latent.value_or(sh.H);
    s.route_words = route_words(h.N) * sh.rows / h.G;
    s.slot_words = u32(h.PCAP / 4);
    s.slots_rows = 2 * h.N;
    s.fslots_rows = std::max<uint32_t>(1, u32(h.fkeys.size())) * h.NF;
    const uint32_t NG = h.N * h.EPC;
    s.plan_outs = {{{1, NG}, {1, NG}, {1, h.rows_cap}, {h.rows_cap / 32, 64}, {1, kPlanWords}}};
    s.arena_bytes = u32(h.arena_bytes);
    return s;
}

CoreRangeSet head_arena_cores(const HeadModel& h) {
    std::vector<CoreCoord> extra = h.rcores;
    for (const auto& c : h.all_ports()) {
        extra.push_back(c);
    }
    extra.push_back(h.watcher);
    return h.engine_cores().merge(cells(extra));
}

tt::tt_metal::WorkloadDescriptor build_head_workload(
    const HeadModel& model,
    const HeadBuffers& buffers,
    const std::vector<ttnn::MeshCoordinate>& coords,
    std::vector<HeadPatches>* patches) {
    // (the arena: an L1 tensor of the arena cores, a shard of at least arena_bytes each - the CBs over its regions bind
    // to it, the kernels take base + offset)
    const Buffer* arena = buffers[SRC_ARENA];
    TT_FATAL(
        arena != nullptr && arena->is_l1() && arena->aligned_size_per_bank() >= uint64_t(model.arena_bytes),
        "fused_head: the arena must be an L1 tensor of at least {} B a core",
        model.arena_bytes);
    TT_FATAL(
        arena->shard_spec().grid().contains(head_arena_cores(model)),
        "fused_head: the arena must span the matmul cores, the router cores, the ports and the watcher");
    tt::tt_metal::WorkloadDescriptor workload;
    if (patches) {
        patches->assign(model.N, HeadPatches{});
    }
    for (const auto& c : coords) {
        ChipProgram cp(model, buffers, {c[0], c[1]});
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(c, c), cp.build(patches ? &(*patches)[c[0] * model.C + c[1]] : nullptr)});
    }
    return workload;
}

void patch_head_program(
    tt::tt_metal::Program& program, const HeadPatches& patches, const HeadBuffers& buffers, HeadLastAddresses& last) {
    // (grouped by source; a source's args only when its address differs from the one this program last got)
    const auto& args = patches.args;
    for (const auto& [i, j] : patches.groups) {
        const uint16_t src = args[i].src;
        const uint32_t addr = buffers.addresses[src];
        auto value = [&](const HeadPatches::Arg& a) -> uint32_t& {
            auto& data = a.common ? tt::tt_metal::GetCommonRuntimeArgs(program, a.kernel)
                                  : tt::tt_metal::GetRuntimeArgs(program, a.kernel, a.core);
            return data[a.index];
        };
        if (last.a[src] != addr) {
            last.a[src] = addr;
            for (size_t k = i; k < j; ++k) {
                value(args[k]) = addr + args[k].offset;
            }
            if (src == SRC_ARENA) {
                const auto cbs = program.circular_buffers();
                for (const auto& cb : patches.arena_cbs) {
                    tt::tt_metal::UpdateDynamicCircularBufferAddress(
                        program, cbs[cb.index]->id(), *buffers[SRC_ARENA], cb.offset);
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head

namespace ttnn::operations::experimental::moe_block::fused_head {

HeadDeviceWeights prepare_head_weights(
    const HeadModel& h,
    const std::vector<float>& gate,
    const std::vector<float>& up,
    const std::vector<float>& down,
    const std::vector<float>& latent_down,
    const std::vector<float>& router_w,
    const std::vector<float>& router_b) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::TensorMemoryLayout;
    const auto& e = *h.eng;
    const auto& sh = h.cfg.shape;
    const uint32_t H = sh.H, Is = sh.I * h.G, Ic = sh.I, G = h.G;
    TT_FATAL(
        gate.size() == size_t(Is) * H && up.size() == gate.size(),
        "fused_head: gate / up must be [Is {}, H {}]",
        Is,
        H);
    TT_FATAL(down.size() == size_t(H) * Is, "fused_head: down must be [H {}, Is {}]", H, Is);
    const uint32_t lat = sh.latent.value_or(0);
    TT_FATAL(latent_down.size() == size_t(lat) * H, "fused_head: latent_down must be [Hr {}, H {}]", lat, H);
    const uint32_t E = sh.E;
    TT_FATAL(
        router_w.size() == size_t(E) * H && router_b.size() == E, "fused_head: the router must be [E {}, H {}]", E, H);
    auto& mesh = *h.mesh;
    const uint32_t banks = u32(mesh.dram_grid_size().x);
    const uint32_t nreg = e.GX * e.GY, per = (nreg + banks - 1) / banks;
    const uint32_t host_rows = per * u32(e.wown_t) * 32, host_cols = banks * 32;
    const tt::tt_metal::MemoryConfig mc{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {banks - 1, 0})),
            {host_rows, 32},
            tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, host_rows, host_cols}),
        tt::tt_metal::TensorLayout(h.cfg.w_dtype, tt::tt_metal::PageConfig(Layout::TILE), mc));
    std::vector<Tensor> shards;
    for (uint32_t dev = 0; dev < h.N; ++dev) {
        const uint32_t r = dev / h.C, c = dev % h.C;
        const uint32_t pos = h.chips.at({r, c}).rg.p;
        // (the chip's gate / up as glu_weight's [H, 2 Ic]: tile column 2 i gate i, 2 i + 1 up i; down [Ic, H]; latent
        // [H, lat / G] - its position's rows of the latent down, transposed)
        auto gu = [&](uint32_t k, uint32_t col) {
            const uint32_t i = col / 64, odd = (col / 32) % 2, b = col % 32;
            const uint32_t row = c * Ic + i * 32 + b;
            return (odd ? up : gate)[size_t(row) * H + k];
        };
        auto wdt = [&](uint32_t k, uint32_t col) { return down[size_t(col) * Is + c * Ic + k]; };
        auto wl = [&](uint32_t k, uint32_t col) { return latent_down[size_t(pos * lat / G + col) * H + k]; };
        std::vector<float> host(size_t(host_rows) * host_cols, 0.0f);
        for (uint32_t cx = 0; cx < e.GX; ++cx) {
            // the column's blocks in stream order: a block = [K tiles][width tiles] of 32 x 32
            std::vector<std::function<float(uint32_t, uint32_t)>> srcs;
            std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
                blocks;  // (src, k0 row tile, kn, c0) + width
            std::vector<uint32_t> widths;
            const uint32_t g0 = 2 * e.h0[cx];
            for (uint32_t pg = 0; pg < e.wg[cx].size(); ++pg) {
                const uint32_t pw = e.wg[cx][pg];
                uint32_t c0 = g0;
                for (uint32_t i = 0; i < pg; ++i) {
                    c0 += e.wg[cx][i];
                }
                uint32_t src = 0;
                if (pg < e.NGU && c0 + pw > 2 * (e.h0[cx] + e.po[cx])) {  // (a dummy pass: the first pair again)
                    c0 = g0;
                }
                if (pg >= e.NGU) {  // (latent passes: this column's slice of the chip's latent columns)
                    src = 2;
                    c0 = e.l0[cx];
                    for (uint32_t i = e.NGU; i < pg; ++i) {
                        c0 += e.wg[cx][i];
                    }
                }
                for (uint32_t j = 0; j < e.NSG; ++j) {
                    blocks.emplace_back(src, j * e.KB, e.KB, c0);
                    widths.push_back(pw);
                }
            }
            for (uint32_t pd = 0; pd < e.wd[cx].size(); ++pd) {
                const uint32_t pw = e.wd[cx][pd];
                const uint32_t c0 = h.rs->column_passes(pos, cx)[pd].c0;
                for (uint32_t i = 0; i < e.NSD; ++i) {
                    blocks.emplace_back(1, i * e.KBD, e.KBD, c0);
                    widths.push_back(pw);
                }
            }
            for (uint32_t cy = 0; cy < e.GY; ++cy) {
                const uint32_t reg = cx * e.GY + cy, bank = reg % banks, slot = reg / banks;
                uint32_t t = 0;  // tile in the region
                for (size_t bi = cy; bi < blocks.size(); bi += e.GY) {
                    const auto [src, k0, kn, c0] = blocks[bi];
                    for (uint32_t kk = 0; kk < kn; ++kk) {
                        for (uint32_t w_ = 0; w_ < widths[bi]; ++w_, ++t) {
                            for (uint32_t a = 0; a < 32; ++a) {
                                const size_t row = size_t(slot * e.wown_t + t) * 32 + a;
                                for (uint32_t b = 0; b < 32; ++b) {
                                    const uint32_t k = (k0 + kk) * 32 + a, col = (c0 + w_) * 32 + b;
                                    host[row * host_cols + bank * 32 + b] =
                                        src == 0 ? gu(k, col) : (src == 1 ? wdt(k, col) : wl(k, col));
                                }
                            }
                        }
                    }
                }
            }
        }
        shards.push_back(Tensor::from_vector(std::move(host), spec));
    }
    HeadDeviceWeights out;
    out.w = ttnn::distributed::from_host_shards(shards, mesh.shape()).to_device(&mesh, mc);
    // the router: W_r [E, H / TP] (the chip's H slice) bf16, the bias and the expert index as [E, 32] tiles
    const uint32_t Hc = H / G;
    const tt::tt_metal::TensorSpec rws(
        ttnn::Shape({E, Hc}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    std::vector<Tensor> rshards;
    for (uint32_t dev = 0; dev < h.N; ++dev) {
        const uint32_t c = dev % h.C;
        std::vector<float> v(size_t(E) * Hc);
        for (uint32_t ee = 0; ee < E; ++ee) {
            std::copy_n(router_w.begin() + size_t(ee) * H + c * Hc, Hc, v.begin() + size_t(ee) * Hc);
        }
        rshards.push_back(Tensor::from_vector(std::move(v), rws));
    }
    out.rw = ttnn::distributed::from_host_shards(rshards, mesh.shape()).to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    std::vector<float> bias(size_t(E) * 32);
    std::vector<uint16_t> idx(size_t(E) * 32);
    for (uint32_t ee = 0; ee < E; ++ee) {
        for (uint32_t b = 0; b < 32; ++b) {
            bias[size_t(ee) * 32 + b] = router_b[ee];
            idx[size_t(ee) * 32 + b] = static_cast<uint16_t>(ee);
        }
    }
    const tt::tt_metal::TensorSpec bs(
        ttnn::Shape({E, 32}),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    const tt::tt_metal::TensorSpec is(
        ttnn::Shape({E, 32}),
        tt::tt_metal::TensorLayout(DataType::UINT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    std::vector<Tensor> bsh(h.N, Tensor::from_vector(bias, bs)), ish(h.N, Tensor::from_vector(idx, is));
    out.rb = ttnn::distributed::from_host_shards(bsh, mesh.shape()).to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    out.rix = ttnn::distributed::from_host_shards(ish, mesh.shape()).to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head
