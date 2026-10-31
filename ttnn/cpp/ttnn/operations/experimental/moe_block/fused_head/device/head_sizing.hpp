// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's host sizing, no device: the pieces head_plan.fit needs to decide whether a layout builds.
//  * RsPlan: the SE down + reduce-scatter's tiling and transport schedule (se_rs.RsPlan over
//    matmul_reduce_scatter.plan_transport).
//  * SeEngineSizing: the matmul cores' arena plan (se_engine3.SeEngine3.__init__ with mesh_device=None).
//  * router_layout / router_min_bytes (router_stream.RouterStream.layout_of / min_bytes).
//  * routing_a2a_sizes / worst_rows / slot_bytes (combine_rf/routing_a2a.py).
//  * routing_paths (head_plan.routing_paths: the routing exchange's senders and forwarding slots).
// Reference tree: mstaletovic/moe-trim, models/demos/moe_block/tt/. Every failure the reference raises inside the
// planner's try block (ValueError / AssertionError / KeyError) is a SizingError carrying the reference's str(err).

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace head_math {
inline int64_t al64(int64_t v) { return (v + 63) / 64 * 64; }
inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace head_math

// A requirement of the build failed (the reference's ValueError / AssertionError / KeyError: what() is its str()).
struct SizingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The matmul cores' arena does not fit its cap at any W ring / h slot count (se_engine3.ArenaTooLarge).
struct ArenaTooLarge : SizingError {
    ArenaTooLarge(const std::string& what, int64_t need, int64_t cap) : SizingError(what), need(need), cap(cap) {}
    int64_t need;
    int64_t cap;
};

// Ordered (name, bytes) regions, in arena order (the reference's dict insertion order).
using Regions = std::vector<std::pair<std::string, int64_t>>;

inline constexpr int64_t kArenaMax = 1344 * 1024;  // se_engine3.ARENA_MAX
inline constexpr uint32_t kDestTiles = 8;          // se_engine3.dest_tiles()

// x_gather3.port_cb_bytes: a port core's static packet CB at K step kb.
int64_t port_cb_bytes(uint32_t kb, uint32_t payload, uint32_t group = 4);
// se_engine3.arena_cap_of: L1 the arena may take from a free-L1 figure.
int64_t arena_cap_of(int64_t l1_free, uint32_t kb, uint32_t payload);

// lin2d.column_split
std::vector<uint32_t> column_split(uint32_t n, uint32_t parts, uint32_t unit);

// ---- matmul_reduce_scatter.plan_transport (mirror off, one wave) ----
enum class RsKind : uint8_t { Line, RingBalanced, RingUnbalanced };
enum class RsDir : uint8_t { Fwd = 0, Bwd = 1, Fin = 2 };
RsKind rs_kind(uint32_t G, bool ring);

struct RsRole {
    RsDir dir;
    bool start;
    uint32_t dist;
};

struct RsChip {
    std::map<std::pair<uint32_t, uint32_t>, RsRole> roles;                 // (block, half)
    std::vector<uint32_t> entries[2];                                      // per half
    std::map<std::pair<uint32_t, RsDir>, std::vector<uint32_t>> arrivals;  // (half, fwd | bwd)
};

struct RsTransport {
    std::vector<RsChip> chips;
    uint32_t slots[2] = {1, 1};  // fwd, bwd
};
RsTransport plan_transport(uint32_t G, RsKind kind);

// ---- se_rs.RsPlan (the parts the sizing reads) ----
class RsPlan {
public:
    struct Pass {
        uint32_t c0;
        uint32_t pw;
        std::optional<uint32_t> entry;
    };
    enum class Op : uint8_t { Pass, Hand, Fin };

    RsPlan(
        uint32_t G,
        bool ring,
        uint32_t GX,
        uint32_t GY,
        uint32_t MT,
        uint32_t NTH,
        int scatter,
        uint32_t nv,
        uint32_t pwmax,
        bool compact);

    uint32_t G, GX, GY, MT, NTH, nv, pwmax, CR, ncores;
    int scatter;
    bool compact;
    RsKind kind;
    RsTransport transport;
    uint32_t P = 0;
    // scatter -1
    uint32_t NB = 0, npb = 0;
    std::vector<uint32_t> w, s;
    // scatter -2
    uint32_t RB = 0, npass = 0;
    std::vector<uint32_t> pn, n0;
    std::optional<uint32_t> hcol;
    bool pm = false;

    uint32_t k(uint32_t cx, uint32_t cy) const { return cy * GX + cx; }
    uint32_t half(uint32_t k) const;
    std::pair<uint32_t, uint32_t> rows_in(uint32_t cy, uint32_t b) const;
    uint32_t count(uint32_t cx, uint32_t cy, uint32_t b) const;
    const std::vector<uint32_t>& entries(uint32_t p, uint32_t h = 0) const { return transport.chips[p].entries[h]; }
    const RsRole& role(uint32_t p, uint32_t b, uint32_t h) const;
    std::vector<Pass> column_passes(uint32_t p, uint32_t cx) const;
    uint32_t finb_tiles(uint32_t cx, uint32_t cy) const;
    // (offsets by entry, HOLD tiles used)
    std::pair<std::map<uint32_t, uint32_t>, uint32_t> hold_offs(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<Op, uint32_t>> ops(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<uint32_t, RsDir>> handoffs(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::optional<uint32_t> down(uint32_t p, RsDir d) const;
    std::optional<uint32_t> piece_entry(uint32_t p, uint32_t h, RsDir d) const;
    std::vector<std::pair<uint32_t, uint32_t>> zone_slots(uint32_t p, uint32_t cx, uint32_t cy, RsDir d) const;
    uint32_t zone_tiles(RsDir d) const;

private:
    bool needs_hold(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const;
};

// ---- se_engine3.SeEngine3 host sizing ----
enum class WeightDtype : uint8_t { Bf4, Bf8, Bf16 };

struct SeEngineArgs {
    uint32_t rows = 0, H = 0, I = 0;
    uint32_t GX = 11, GY = 10, MT = 1, KB = 16;
    uint32_t shards = 1;
    const RsPlan* rs = nullptr;
    std::optional<uint32_t> latent;  // tiles a chip
    bool prep = false;
    std::optional<std::pair<uint32_t, uint32_t>> router;  // (ET, KTO): the logits on the engine
    int64_t arena_max = kArenaMax;
    WeightDtype w_dtype = WeightDtype::Bf8;
    uint32_t xring = 0;
};

struct SeEngineSizing {
    explicit SeEngineSizing(const SeEngineArgs& a);  // throws SizingError / ArenaTooLarge

    uint32_t NCH, KT, HT, NTH, KB, NSG, SPS, S0, S1, NPG, NPD, KBD, NSD, NB, WSLOT, wtb;
    uint32_t POMAX, PWDMAX, LWMAX = 0, WRING = 0, HSL = 1;
    std::vector<uint32_t> po;
    std::vector<std::vector<uint32_t>> wg, wd;
    int64_t arena_bytes = 0;
    Regions off;    // region offsets, arena order
    Regions sizes;  // each region's bytes (overlays share their host's)

    int64_t dram_w_bytes() const { return int64_t(KT * 2 * HT + HT * NTH) * wtb; }
};

// ---- router_stream.RouterStream ----
// (offsets, bytes) of the router cores' arena regions at K block kbr.
std::pair<Regions, int64_t> router_layout(uint32_t G, uint32_t ET, uint32_t K, uint32_t kbr, uint32_t ETF = 0);
int64_t router_min_bytes(uint32_t G, uint32_t E, uint32_t K, bool split);

// ---- combine_rf/routing_a2a.py ----
int64_t routing_header_bytes(uint32_t EPC, uint32_t M);
int64_t routing_slot_bytes(uint32_t S, uint32_t K, uint32_t EPC, uint32_t M);
int64_t routing_worst_rows(uint32_t N, uint32_t S, uint32_t K, uint32_t EPC);
struct RoutingSizes {
    uint32_t M;
    int64_t inst_cap, plan_l1, port_l1, packet, slots_dram, local;
    bool fits;
    std::string why;
};
RoutingSizes routing_a2a_sizes(
    uint32_t N, uint32_t S, uint32_t K, uint32_t EPC, uint32_t A, int64_t rows_cap, uint32_t M = 0, uint32_t TPP = 32);

// ---- head_plan.routing_paths ----
struct RoutingPaths {
    uint32_t nx;     // the most packets an X sender sends
    uint32_t ny;     // ... a Y sender
    uint32_t nf;     // ... a port forwards
    uint32_t nsend;  // the most senders a chip
};
// The routing exchange's paths on an R x C mesh, or why they do not build.
std::variant<RoutingPaths, std::string> routing_paths(uint32_t R, uint32_t C, bool ring, bool wrap_y, uint32_t links);

// Python-compatible "%.<p>f" (exact-value rounding, ties to even) and list repr.
std::string fmt_fixed(double v, int precision);
std::string py_list(const std::vector<uint32_t>& v);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
