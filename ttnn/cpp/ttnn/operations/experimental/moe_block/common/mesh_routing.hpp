// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math shared by the MoE block's ops: mesh / torus coordinates, dimension-ordered routes (Y first, then X) with
// the fabric's tie rule, the dispatch's route-word and span tables, the manifest ring layout, the port LUTs.

#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "moe_block_config.hpp"

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

// (mesh row, mesh column); X moves along a row (column index), Y along a column (row index).
struct ChipCoord {
    uint32_t r = 0;
    uint32_t c = 0;
    auto operator<=>(const ChipCoord&) const = default;
};

struct Hop {
    ChipCoord chip;  // the chip the hop leaves
    Dir dir;
    bool operator==(const Hop&) const = default;
};

// An R x C mesh; an axis wraps (a ring) only where the fabric is a torus on it AND the side is > 2.
struct MeshTopology {
    uint32_t R = 1;
    uint32_t C = 1;
    bool wrap_x = false;
    bool wrap_y = false;

    // torus = (X wraps, Y wraps) as the fabric is cabled; applies the side > 2 rule (sparse_all_gather_2d._mesh_plan).
    static MeshTopology of(uint32_t R, uint32_t C, bool torus_x, bool torus_y);
    uint32_t num_chips() const { return R * C; }
    uint32_t index(ChipCoord ch) const { return ch.r * C + ch.c; }
    ChipCoord coord(uint32_t idx) const { return {idx / C, idx % C}; }
    std::optional<ChipCoord> neighbor(ChipCoord ch, Dir d) const;
    bool operator==(const MeshTopology&) const = default;
};

// [(chip, direction)] hops from src to home: Y first, then X (lines: forced; rings: the shorter way). A ring tie (half
// a ring away) goes as the fabric's routing tables send it: by the parity of the source's position on that axis, even
// Y- / X+, odd Y+ / X- (sparse_combine_l1.route, tie_mode 2; its `link` argument is unused there).
std::vector<Hop> route(ChipCoord src, ChipCoord home, const MeshTopology& topo);
// The first hop's direction, nullopt for src == dst.
std::optional<Dir> first_hop(ChipCoord src, ChipCoord dst, const MeshTopology& topo);
// The link of every row from chip src to accumulator acc of chip home.
inline uint32_t link_of(uint32_t acc, uint32_t src, uint32_t home, uint32_t L) { return (acc + src + home) % L; }

// Route words per own row: 1 up to 16 chips (mask | order << 16 | link << 17), 2 beyond (mask, order | link << 1).
inline uint32_t route_words(uint32_t N) { return N <= 16 ? 1 : 2; }
// (a, b) packed a | b << 4: hops forward / backward from pos on an n-line / n-ring covering tmask's set bits.
uint8_t span_entry(uint32_t n, bool wrap, uint32_t pos, uint32_t tmask, uint32_t tie);
// [pos][tmask][tie] flattened: n x 2^n x 2 entries.
std::vector<uint8_t> span_table(uint32_t n, bool wrap);
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
// Per position: the most hops a tree's span goes forward / backward from it.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> axis_reach(uint32_t n, bool wrap);
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
MeshPlan mesh_plan(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t num_links, bool torus_x, bool torus_y);
// Host-only: every neighbour connected by links 0..num_links-1 (tests, planners without a device).
MeshPlan mesh_plan_uniform(const MeshTopology& topo, uint32_t num_links);

// Port LUTs: per chip lut[direction * 2 + link] = port id (the chip's ports in DIRS / link order), 0xFF
// for none; and the chip's port count.
struct PortLuts {
    std::map<ChipCoord, std::array<uint8_t, 8>> lut;
    std::map<ChipCoord, uint32_t> num_ports;
};
PortLuts port_luts(const MeshPlan& plan);

}  // namespace ttnn::operations::experimental::moe_block
