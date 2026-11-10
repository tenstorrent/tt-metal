// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math shared by the MoE block's ops: mesh / torus coordinates, the fabric's first hop between every pair of
// chips (FirstHops: asked of the fabric, never computed here), the dispatch's route-word and span tables, the manifest
// ring layout, the port LUTs.

#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

// Port directions in the reference's DIRS order; the value is the DIRS index (kernels and LUTs use it).
enum class Dir : uint8_t { XPlus = 0, XMinus = 1, YPlus = 2, YMinus = 3 };
inline constexpr std::array<Dir, 4> kDirs = {Dir::XPlus, Dir::XMinus, Dir::YPlus, Dir::YMinus};
std::string_view dir_name(Dir d);  // "X+", "X-", "Y+", "Y-"
Dir opposite(Dir d);
inline bool is_x(Dir d) { return d == Dir::XPlus || d == Dir::XMinus; }
inline bool is_plus(Dir d) { return d == Dir::XPlus || d == Dir::YPlus; }

// The block's chip coordinate (SP row, TP column); X moves along an SP row (TP index), Y along a TP column (SP index).
// With SP on mesh axis 0 (the default) it is the mesh coordinate (row, column); with SP on mesh axis 1
// (MeshTopology.transposed: the 4x4 tray-half stage, SP on the axis-1 ring) it is the mesh coordinate (column, row):
// r = the mesh column, c = the mesh row. Every op works in these coordinates; only mesh_coord / chip_of cross over.
struct ChipCoord {
    uint32_t r = 0;
    uint32_t c = 0;
    auto operator<=>(const ChipCoord&) const = default;
};

// An R (SP) x C (TP) mesh in the block's coordinates; an axis wraps (a ring) only where the fabric is a torus on it AND
// the side is > 2. transposed: SP on mesh axis 1 (the block's Y is the mesh's axis 1, its X the mesh's axis 0).
struct MeshTopology {
    uint32_t R = 1;
    uint32_t C = 1;
    bool wrap_x = false;
    bool wrap_y = false;
    bool transposed = false;

    // R, C, torus = the block's SP and TP sides and (X wraps, Y wraps) in the block's coordinates; applies the side > 2
    // rule (sparse_all_gather_2d._mesh_plan).
    static MeshTopology of(uint32_t R, uint32_t C, bool torus_x, bool torus_y, bool transposed = false);
    // From the mesh's shape and the fabric's torus axes (mesh axis 1 = the fabric's X, axis 0 its Y) for SP on
    // sp_axis.
    static MeshTopology of_mesh(
        uint32_t mesh_rows, uint32_t mesh_cols, bool fabric_torus_x, bool fabric_torus_y, uint32_t sp_axis);
    uint32_t num_chips() const { return R * C; }
    uint32_t index(ChipCoord ch) const { return ch.r * C + ch.c; }
    ChipCoord coord(uint32_t idx) const { return {idx / C, idx % C}; }
    std::optional<ChipCoord> neighbor(ChipCoord ch, Dir d) const;
    bool operator==(const MeshTopology&) const = default;
};

// The mesh coordinate of a chip, and the chip of a mesh coordinate (the transpose with SP on mesh axis 1).
tt::tt_metal::distributed::MeshCoordinate mesh_coord(ChipCoord ch, const MeshTopology& topo);
ChipCoord chip_of(const tt::tt_metal::distributed::MeshCoordinate& coord, const MeshTopology& topo);
// Per-chip host shards given in the block's chip order (index r C + c) reordered to the mesh's row-major order (what
// from_host_shards / a mesh mapper take): the identity unless transposed.
template <typename T>
std::vector<T> to_mesh_order(std::vector<T> shards, const MeshTopology& topo) {
    if (!topo.transposed) {
        return shards;
    }
    std::vector<T> out;
    out.reserve(shards.size());
    for (uint32_t mr = 0; mr < topo.C; ++mr) {      // mesh rows = the block's TP columns
        for (uint32_t mc = 0; mc < topo.R; ++mc) {  // mesh columns = the block's SP rows
            out.push_back(shards.at(topo.index(ChipCoord{mc, mr})));
        }
    }
    return out;
}

// The fabric's first hop from every chip to every other, as its routing tables send a unicast to that chip - asked of
// the fabric (tt_fabric::get_eth_forwarding_direction), never derived: its dimension order and how it splits a ring
// tie (a chip half a ring away) are the fabric's own. Every op that picks the port a fabric packet leaves by reads
// this (the combine's ports, the MoE head's routing exchange and its planner kernel). A fabric direction is matched
// to the block's by asking the fabric the way to each of the chip's neighbours.
struct FirstHops {
    uint32_t N = 0;               // chips (topo.num_chips())
    std::vector<uint8_t> dirs;    // [src * N + dst]: the Dir (its DIRS index); src == dst: 0, unused
    Dir at(uint32_t src, uint32_t dst) const { return static_cast<Dir>(dirs.at(static_cast<size_t>(src) * N + dst)); }
    Dir at(ChipCoord src, ChipCoord dst, const MeshTopology& topo) const { return at(topo.index(src), topo.index(dst)); }
    // chip src's row for a kernel: 16 directions a word, 2 bits each (dst d at bits 2 (d % 16) of word d / 16)
    std::vector<uint32_t> packed_row(uint32_t src) const;
    bool operator==(const FirstHops&) const = default;
};
inline constexpr uint32_t kFirstHopsPerWord = 16;
FirstHops fabric_first_hops(const tt::tt_metal::distributed::MeshDevice& mesh_device, const MeshTopology& topo);
// The link of every row from chip src to accumulator acc of chip home.
inline uint32_t link_of(uint32_t acc, uint32_t src, uint32_t home, uint32_t L) { return (acc + src + home) % L; }

// Route words per own row: 1 up to 16 chips (mask | order << 16 | link << 17), 2 beyond (mask, order | link << 1).
inline uint32_t route_words(uint32_t N) { return N <= 16 ? 1 : 2; }
// (a, b) packed a | b << 4: hops forward / backward from pos on an n-line / n-ring covering tmask's set bits.
uint8_t span_entry(uint32_t n, bool wrap, uint32_t pos, uint32_t tmask, uint32_t tie);
// The kernels' ring span table (span_entry from position 0 by rotated mask and tie); {0} for a line.
std::vector<uint8_t> ring_table(uint32_t n, bool wrap);
// The compile-time span constants of the dispatch kernels (span_subs: __WRAPX__ __WRAPY__ __XRING__ __YRING__).
struct SpanTables {
    bool wrap_x = false;
    bool wrap_y = false;
    std::vector<uint8_t> xring;  // ring_table(C, wrap_x)
    std::vector<uint8_t> yring;  // ring_table(R, wrap_y)
};
SpanTables span_subs(uint32_t R, uint32_t C, bool wrap_x, bool wrap_y);
// (ring 1's offset, ring 2's offset, total) in 16-byte entries of a port core's manifest (S own rows a chip).
struct ManifestLayout {
    uint32_t ring1_offset = 0;
    uint32_t ring2_offset = 0;
    uint32_t total_entries = 0;
};
ManifestLayout manifest_layout(uint32_t R, uint32_t C, bool wrap_x, bool wrap_y, uint32_t S);

// Ports of every chip (sparse_all_gather_2d.mesh_plan): per chip, per direction with a neighbour, the peer and the
// fabric link indices used (the first num_links of get_forwarding_link_indices).
struct MeshPort {
    ChipCoord peer;
    std::vector<uint32_t> links;
    bool operator==(const MeshPort&) const = default;
};
struct MeshPlan {
    MeshTopology topo;
    uint32_t num_links = 0;
    std::map<ChipCoord, std::array<std::optional<MeshPort>, 4>> chips;  // [chip][DIRS index]
    bool operator==(const MeshPlan&) const = default;
};
// From the fabric: throws if a neighbour is not a direct link or has fewer than num_links links.
// torus_x / torus_y: the fabric's (mesh axis 1 / axis 0); sp_axis: the mesh axis SP runs on.
MeshPlan mesh_plan(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t num_links,
    bool torus_x,
    bool torus_y,
    uint32_t sp_axis = 0);
// Host-only: every neighbour connected by links 0..num_links-1 (tests, planners without a device).
MeshPlan mesh_plan_uniform(const MeshTopology& topo, uint32_t num_links);

// Port LUTs: per chip lut[direction * kMaxLinks + link] = port id (the chip's ports in DIRS / link order), 0xFF
// for none; and the chip's port count. Every op numbers a chip's ports this way (the MoE head, the combine).
inline constexpr uint32_t kMaxLinks = 2;  // links per direction a LUT has room for
struct PortLuts {
    std::map<ChipCoord, std::array<uint8_t, 4 * kMaxLinks>> lut;
    std::map<ChipCoord, uint32_t> num_ports;
};
// TT_FATAL if plan.num_links > kMaxLinks.
PortLuts port_luts(const MeshPlan& plan);

}  // namespace ttnn::operations::experimental::moe_block
