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
// A requirement a layout does not meet is a SizingError saying which one, with its numbers.

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace head_math {
inline int64_t al64(int64_t v) { return (v + 63) / 64 * 64; }
inline int64_t al16(int64_t v) { return (v + 15) / 16 * 16; }
inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace head_math

// A requirement of the build failed (what(): which one, with its numbers).
struct SizingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The matmul cores' arena does not fit its cap at any W ring / h slot count (se_engine3.ArenaTooLarge).
struct ArenaTooLarge : SizingError {
    ArenaTooLarge(const std::string& what, int64_t need, int64_t cap) : SizingError(what), need(need), cap(cap) {}
    int64_t need;
    int64_t cap;
};

// Ordered (name, bytes) regions, in arena order.
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

    // ---- the device build's tables (se_rs.RsPlan) ----
    struct Arrival {
        uint32_t entry, slot, threshold;
    };
    struct PortUnit {
        uint32_t entry, block, mask, slot0, slot1;
        bool pieces;
    };
    uint32_t link(uint32_t k) const { return k % nv; }
    std::optional<uint32_t> up(uint32_t p, RsDir d) const;
    // per direction (fwd, bwd): [(entry, zone slot, threshold)]
    std::array<std::vector<Arrival>, 2> arrivals(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<PortUnit> port_units(uint32_t p, RsDir d) const;
    std::vector<RsDir> finb_readers(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<uint32_t, uint32_t>> port_cores(uint32_t l) const;  // (cx, cy) of link l, k order
    std::tuple<uint32_t, uint32_t, uint32_t> fin_map(uint32_t p, uint32_t cx, uint32_t cy) const;
    uint32_t prefix_mask(uint32_t q, uint32_t h, RsDir d, uint32_t slot) const;
    uint32_t counts_word(uint32_t cx, uint32_t cy) const;
    uint32_t finb_off(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const;
    std::vector<uint32_t> pass_widths(uint32_t cx) const;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> pass_bases(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<uint32_t> compute_args(uint32_t p, uint32_t cx, uint32_t cy) const;

private:
    bool needs_hold(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const;
};

// ---- se_engine3.SeEngine3 host sizing ----
enum class WeightDtype : uint8_t { Bf4, Bf8, Bf16 };

struct SeEngineArgs {
    uint32_t rows = 0, H = 0, I = 0;
    uint32_t GX = 11, GY = 10, MT = 1, KB = 16;
    uint32_t shards = 1;
    const RsPlan* rs = nullptr;                           // the down + reduce-scatter's plan (required)
    std::optional<uint32_t> latent;                       // tiles a chip
    std::optional<std::pair<uint32_t, uint32_t>> router;  // (ET, KTO): the logits on the engine
    int64_t arena_max = kArenaMax;
    WeightDtype w_dtype = WeightDtype::Bf8;
    uint32_t xring = 0;
};

struct SeEngineSizing {
    explicit SeEngineSizing(const SeEngineArgs& a);  // throws SizingError / ArenaTooLarge

    SeEngineArgs args;  // (args.rs: not owned - valid while the caller's RsPlan is)
    uint32_t GX, GY, MT, CR, rows, H, I, shards;
    uint32_t NCH, KT, HT, NTH, KB, NSG, SPS, S0, S1, NPG, NGU, NPD, KBD, NSD, NB, WSLOT, wtb;
    uint32_t POMAX, PWDMAX, PGW, LWMAX = 0, LT = 0, WRING = 0, HSL = 1, XR = 0, PFULL = 0, PBCT = 0;
    int64_t LATP = 0, xown_t = 0, wown_t = 0, acc_pages = 0;
    bool xring = false, xfree = false;
    std::vector<uint32_t> fam0, fam1;  // the x families' owner columns
    std::vector<uint32_t> po, h0, n0d, lw, l0;
    std::vector<std::vector<uint32_t>> wg, wd;
    // the router logits on the engine (args.router)
    uint32_t ETC = 0, RG = 0, KC = 0;
    bool rtr_zones = false, rtr_latz = false, rtr_overlay = false;
    Regions rs_sizes;  // the reduce-scatter's regions (se_engine3._rs_regions)
    int64_t arena_bytes = 0;
    Regions off;    // region offsets, arena order
    Regions sizes;  // each region's bytes (overlays share their host's)
    std::vector<int64_t> seq_ends;  // the sequential regions' bounds (the CB ring check's)
    // compute kernel groups: columns with the same (gate / up widths, down widths), in first-column order
    std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>, std::vector<uint32_t>>> groups;

    int64_t dram_w_bytes() const { return int64_t(KT * 2 * HT + HT * NTH) * wtb; }
    int64_t at(const std::string& region) const;  // off[region]
    bool in_fam0(uint32_t c) const;
    bool in_fam1(uint32_t c) const;
    // (owner column, family, index in the family) of x step j
    std::tuple<uint32_t, uint32_t, uint32_t> owner(uint32_t j) const;
    // chunk m's local row tile lr -> the global row tile
    uint32_t grow(uint32_t m, uint32_t lr) const;
    // (first local row, rows) of core row cy's MT row tiles in owner block b (latent / prep)
    std::pair<uint32_t, uint32_t> la_rows(uint32_t cy, uint32_t b) const;
    uint32_t bsize(uint32_t c, uint32_t b) const;
    int64_t lrdy_bytes() const { return 16 * int64_t(std::max<uint32_t>(4, NCH)); }
    bool fin_rows() const;  // se_engine3._fin_rows
};

// ---- router_stream.RouterStream ----
// (offsets, bytes) of the router cores' arena regions at K block kbr.
std::pair<Regions, int64_t> router_layout(uint32_t G, uint32_t ET, uint32_t K, uint32_t kbr, uint32_t ETF = 0);
int64_t router_min_bytes(uint32_t G, uint32_t E, uint32_t K);

// ---- combine_rf/routing_a2a.py ----
int64_t routing_header_bytes(uint32_t EPC, uint32_t M);
int64_t routing_slot_bytes(uint32_t S, uint32_t K, uint32_t EPC, uint32_t M);
int64_t routing_worst_rows(uint32_t N, uint32_t S, uint32_t K, uint32_t EPC);
struct RoutingSizes {
    uint32_t M;
    int64_t inst_cap, plan_l1, port_l1;
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

// "%.<p>f" of v, and "[a, b, ...]" of a list (the planner's messages).
std::string fmt_fixed(double v, int precision);
std::string list_str(const std::vector<uint32_t>& v);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
