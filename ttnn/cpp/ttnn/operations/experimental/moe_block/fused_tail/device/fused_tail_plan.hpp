// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host plans of the fused tail: the pure parts (schedules, geometry, splits), and the per-resources plans
// (TailEnginePlan, TailA2APlan) that add the device-dependent parts (port cores, fabric ties), built once per
// MoEBlockResources.
//
// A tail's TP group is a mesh row (cluster_axis 1): position p = the chip's column, G = the mesh's columns. Fwd sends
// toward p + 1 (X+), Bwd toward p - 1 (X-); a ring when the row wraps (torus X and G > 2).

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::fused_tail {

enum class TailDir : uint8_t { Fwd = 0, Bwd = 1 };
inline TailDir opposite(TailDir d) { return d == TailDir::Fwd ? TailDir::Bwd : TailDir::Fwd; }
inline Dir mesh_dir(TailDir d) { return d == TailDir::Fwd ? Dir::XPlus : Dir::XMinus; }
// The TP position next to p toward d (nullopt at a line's end).
std::optional<uint32_t> row_neighbor(uint32_t p, uint32_t G, bool ring, TailDir d);

// A block of the gather: the TP position it is from and its part (0 whole; 1 / 2 the first / second half of the
// columns or banks - a ring's opposite block split between the two directions).
struct Unit {
    uint32_t pos = 0;
    uint32_t part = 0;
    auto operator<=>(const Unit&) const = default;
};

// The gather's schedule: (units sent toward p + 1, units sent toward p - 1), own first.
std::pair<std::vector<Unit>, std::vector<Unit>> gather_schedule(uint32_t p, uint32_t G, bool ring);
// Chip p's chunk positions -> TP positions (own first, then by hop distance).
std::vector<uint32_t> chunk_order(uint32_t p, uint32_t G, bool ring);
// n into `parts` contiguous ranges, as even as possible.
std::vector<std::pair<uint32_t, uint32_t>> even_split(uint32_t n, uint32_t parts);
// lin2d.column_split: per-column tiles, multiples of `unit`, as even as possible.
std::vector<uint32_t> column_split(uint32_t n_tiles, uint32_t columns, uint32_t unit);
// se_engine3._ring_pages: a FIFO CB's pages so no pushed block straddles its end.
uint32_t ring_pages(uint32_t pages, const std::vector<uint32_t>& pushes);

// ---- TailA2A ----
inline constexpr uint32_t kIncEvery = 8;  // a fused arrival increment every 8 runs of a destination (and its last)

// tail_a2a.TailSchedule: who sends what to whom. Ports are (position, direction, link).
class TailSchedule {
public:
    using Port = std::tuple<uint32_t, TailDir, uint32_t>;
    TailSchedule() = default;
    // ties: (p, q) at distance G / 2 on a ring -> the fabric's first direction (default Fwd).
    TailSchedule(
        uint32_t G,
        bool ring,
        uint32_t L,
        uint32_t NTB,
        uint32_t RT,
        uint32_t NBK,
        uint32_t RP,
        std::map<std::pair<uint32_t, uint32_t>, TailDir> ties = {});

    TailDir dir(uint32_t p, uint32_t q) const;
    std::vector<Unit> dests(uint32_t p, TailDir d) const;
    std::vector<std::pair<uint32_t, uint32_t>> run_list(uint32_t part, uint32_t l) const;  // [(first tile, pages)]
    uint32_t runs(uint32_t part, uint32_t l) const { return static_cast<uint32_t>(run_list(part, l).size()); }
    uint32_t hops(uint32_t p, uint32_t q, TailDir d) const;
    // Arrival increments a port expects in a call, and the ports a port sends a ready to.
    uint32_t expect(const Port& port) const;
    std::vector<Port> readies(const Port& port) const;  // sorted

    uint32_t G = 0, L = 0, NTB = 0, RT = 0, NBK = 0, RP = 0;
    bool ring = false;
    std::map<std::pair<uint32_t, uint32_t>, TailDir> ties;

private:
    std::map<Port, uint32_t> expect_;
    std::map<Port, std::set<Port>> readies_;
};

// Shapes of a TailA2A call (TailA2A.__init__).
struct TailA2AGeometry {
    uint32_t G = 0;      // TP
    uint32_t rows = 0;   // a mesh row's tokens (S/SP)
    uint32_t H = 0;      // emb_dim
    uint32_t S = 0;      // owned rows a chip
    uint32_t RT = 0;     // owned row tiles
    uint32_t NTH = 0;    // H / 32
    uint32_t NTB = 0;    // a destination block's tiles (H / 32 / G)
    uint32_t W = 0;      // tile columns a worker
    uint32_t NW = 0;     // workers
    uint32_t NBK = 0;    // DRAM banks
    uint32_t RP = 0;     // pages a run (fabric payload / page)
    uint32_t group = 0;  // runs a port group

    static TailA2AGeometry make(uint32_t G, uint32_t rows, uint32_t H, uint32_t NBK, size_t max_payload_bytes);
};

// Everything a TailA2A program needs that does not change between calls (built once per resources).
struct TailA2APlan {
    TailA2AGeometry geom;
    bool ring = false;
    uint32_t num_links = 0;
    TailSchedule sched;
    // port core of (chip, direction, link); only directions with a neighbour
    std::map<std::tuple<ChipCoord, TailDir, uint32_t>, tt::tt_metal::CoreCoord> ports;
    tt::tt_metal::CoreRangeSet port_union;  // every chip's port cores (the persistent words live there)
    std::vector<tt::tt_metal::CoreCoord> workers;
    std::map<std::pair<ChipCoord, TailDir>, std::vector<uint32_t>> links;  // fabric link ids toward each neighbour
};

std::shared_ptr<const TailA2APlan> make_tail_a2a_plan(MoEBlockResources& resources, uint32_t rows, uint32_t H);

// ---- TailEngine ----
inline constexpr uint32_t kXT = 4;  // a step's exchange: up to 3 partial tiles + the half's sum of squares
inline constexpr uint32_t kArenaMax = 1400 * 1024;
inline constexpr uint32_t kLandGroup = 3;  // WG: one fused landed increment a 3 columns' shares
inline constexpr uint32_t kPortGroup = 8;  // PG: packets a port group

// TailEngine.__init__'s geometry (rows of a mesh row, Hr latent, Ho = H / TP, chunks = TP, GX x GY engine).
struct TailEngineGeometry {
    uint32_t rows = 0, Hr = 0, Ho = 0, chunks = 0, GX = 0, GY = 0, wtb = 0;
    uint32_t KT = 0, NTO = 0, RT = 0, PAIRS = 0, NSTMAX = 0, NOWNMAX = 0, KH = 0, PNMAX = 0, FNR = 0, SHMAX = 0;
    uint32_t NSLOT = 0;
    std::vector<uint32_t> nst, nown, pn, n0, FNS;
    std::vector<std::pair<uint32_t, uint32_t>> kh, wk, ak;
    std::vector<std::array<std::pair<uint32_t, uint32_t>, 2>> fin;  // per column: (fo, fn) of half 0 / 1
    uint32_t arena_bytes = 0;

    // cap_bytes: the L1 the engine's CBs may take; NSLOT is the most ring slots that fit (>= min(2, NSTMAX)).
    static TailEngineGeometry make(
        uint32_t rows,
        uint32_t Hr,
        uint32_t Ho,
        uint32_t chunks,
        uint32_t GX,
        uint32_t GY,
        uint32_t wtb,
        uint32_t cap_bytes);
    // TailEngine._layout: (region offsets in the reference's order, total), 64 B aligned.
    std::pair<std::vector<uint32_t>, uint32_t> layout(uint32_t nslot) const;
    // Tiles in one W' region (a half column's K range, padded to the largest).
    uint32_t region_tiles() const;
};

// TailEngine._relays: the watchers of chip p: {(cx, cy): [s | c0 << 8 | c1 << 16]} (sorted).
std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> engine_relays(
    uint32_t p, uint32_t G, bool ring, const TailEngineGeometry& g, uint32_t wg = kLandGroup);

struct TailEnginePlan {
    TailEngineGeometry geom;
    bool ring = false;
    uint32_t num_links = 0;
    uint32_t nbank = 0;  // DRAM banks the W' regions are spread over (dram_grid_size().x)
    uint32_t RP = 0;     // tiles a gather packet
    float norm_eps = 0;  // the latent rmsnorm's eps
    std::map<std::pair<ChipCoord, TailDir>, std::vector<uint32_t>> links;  // fabric link ids toward each neighbour
    // the four gather ports (direction, link), at fixed cells in the column right of the engine (every chip)
    std::map<std::pair<TailDir, uint32_t>, tt::tt_metal::CoreCoord> ports;
    tt::tt_metal::CoreRangeSet engine_cores;
    tt::tt_metal::CoreRangeSet port_cores;
};

// cap_bytes: L1 the engine may use (at most kArenaMax).
std::shared_ptr<const TailEnginePlan> make_tail_engine_plan(
    MoEBlockResources& resources, uint32_t rows, uint32_t Hr, uint32_t Ho, uint32_t wtb, uint32_t cap_bytes);

// The fabric link ids toward each row neighbour (MeshPlan's X+ / X- ports).
std::map<std::pair<ChipCoord, TailDir>, std::vector<uint32_t>> row_links(const MeshPlan& plan);

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
