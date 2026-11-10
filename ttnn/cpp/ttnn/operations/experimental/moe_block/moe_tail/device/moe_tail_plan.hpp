// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// moe_tail's host plans: everything about a tail call that depends only on the model's dims, the mesh and the fabric,
// computed once per MoEBlockResources and reused by every call (moe_tail.cpp caches them with resources.plan()).
//
// What a plan decides - the important output is WHO MOVES WHICH TILES WHERE:
//  * all-to-all (no latent - GLM-5.3, K2.7, DSv4.1-Flash, ...): each chip holds its owned rows of the full hidden dim
//    and must end with every row of its TP column block. The plan cuts the rows into tile blocks per destination chip
//    (TailAllToAllGeometry: worker cores and their tile columns, the fabric run size), and TailSchedule decides which
//    port (direction, link) of which chip carries which destination's block, in runs of DRAM-bank-contiguous pages,
//    and how many arrival increments / readies each port must see (the fence the kernels wait on).
//  * latent engine (Kimi-K3): the latent rows are all-gathered over TP straight into an engine of
//    engine_columns x engine_rows cores, then rmsnorm + the up projection. TailEngineGeometry splits the latent's K
//    into two halves (a core pair per row tile, one half each), the output columns over the engine columns, the call's
//    row tiles into steps round-robin over the pairs, and sizes the L1 rings (ring_slots: the most that fit the free
//    L1). gather_schedule / chunk_order / engine_watchers decide which blocks each gather port sends and which core
//    counts a landed step for its column group.
//
// Main functions: make_tail_all_to_all_plan and make_tail_engine_plan (the per-resources plans, with the port and
// worker cores placed on the device's grid); TailAllToAllGeometry::make and TailEngineGeometry::make (pure sizing);
// TailSchedule (the all-to-all's routing and fence counts); gather_schedule, chunk_order, engine_watchers (the engine's
// all-gather). The program factory (moe_tail_program_factory.cpp) turns a plan into kernels' args.
//
// A tail's TP group is a mesh row (cluster_axis 1): position p = the chip's column, tp_size = the mesh's columns.
// Forward sends toward p + 1 (X+), Backward toward p - 1 (X-); a ring when the row wraps (torus X and tp_size > 2).

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

namespace ttnn::operations::experimental::moe_block::moe_tail {

enum class TailDirection : uint8_t { Forward = 0, Backward = 1 };
inline TailDirection opposite(TailDirection direction) {
    return direction == TailDirection::Forward ? TailDirection::Backward : TailDirection::Forward;
}
inline Dir mesh_direction(TailDirection direction) {
    return direction == TailDirection::Forward ? Dir::XPlus : Dir::XMinus;
}
// The TP position next to `position` toward `direction` (nullopt at a line's end).
std::optional<uint32_t> row_neighbor(uint32_t position, uint32_t tp_size, bool ring, TailDirection direction);

// A block of the gather: the TP position it is from and its part (0 whole; 1 / 2 the first / second half of the
// columns or banks - a ring's opposite block split between the two directions).
struct Unit {
    uint32_t position = 0;
    uint32_t part = 0;
    auto operator<=>(const Unit&) const = default;
};

// The gather's schedule: (units sent toward position + 1, units sent toward position - 1), own first.
std::pair<std::vector<Unit>, std::vector<Unit>> gather_schedule(uint32_t position, uint32_t tp_size, bool ring);
// The chunk positions of the chip at `position` -> TP positions (own first, then by hop distance).
std::vector<uint32_t> chunk_order(uint32_t position, uint32_t tp_size, bool ring);
// `count` into `parts` contiguous ranges, as even as possible.
std::vector<std::pair<uint32_t, uint32_t>> even_split(uint32_t count, uint32_t parts);
// Per-column tiles, multiples of `unit`, as even as possible.
std::vector<uint32_t> column_split(uint32_t num_tiles, uint32_t columns, uint32_t unit);
// A FIFO CB's pages so no pushed block straddles its end.
uint32_t ring_pages(uint32_t pages, const std::vector<uint32_t>& pushes);

// ---- all-to-all ----
inline constexpr uint32_t kArrivalIncrementEvery = 8;  // a fused arrival increment every 8 runs of a destination
inline constexpr uint32_t kMaxPortDestinations = 8;    // port_fabric_sender's per-destination run counters

// Who sends what to whom. Ports are (position, direction, link).
class TailSchedule {
public:
    using Port = std::tuple<uint32_t, TailDirection, uint32_t>;
    TailSchedule() = default;
    // ties: (source, destination) at distance tp_size / 2 on a ring -> the fabric's first direction (default Forward).
    TailSchedule(
        uint32_t tp_size,
        bool ring,
        uint32_t num_links,
        uint32_t block_tiles,
        uint32_t owned_row_tiles,
        uint32_t num_dram_banks,
        uint32_t pages_per_run,
        std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties = {});

    TailDirection direction(uint32_t source, uint32_t destination) const;
    std::vector<Unit> destinations(uint32_t source, TailDirection toward) const;
    // [(first tile, pages)] of a block's row tile carried by link `link`
    std::vector<std::pair<uint32_t, uint32_t>> run_list(uint32_t part, uint32_t link) const;
    uint32_t runs(uint32_t part, uint32_t link) const { return static_cast<uint32_t>(run_list(part, link).size()); }
    uint32_t hops(uint32_t source, uint32_t destination, TailDirection toward) const;
    // Arrival increments a port expects in a call, and the ports a port sends a ready to.
    uint32_t expected_arrivals(const Port& port) const;
    std::vector<Port> readies(const Port& port) const;  // sorted

    uint32_t tp_size = 0, num_links = 0, block_tiles = 0, owned_row_tiles = 0, num_dram_banks = 0, pages_per_run = 0;
    bool ring = false;
    std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties;

private:
    std::map<Port, uint32_t> expected_arrivals_;
    std::map<Port, std::set<Port>> readies_;
};

// Shapes of an all-to-all call.
struct TailAllToAllGeometry {
    uint32_t tp_size = 0;          // the TP group's chips
    uint32_t rows = 0;             // a mesh row's tokens (S / SP)
    uint32_t hidden_dim = 0;       // emb_dim
    uint32_t owned_rows = 0;       // rows a chip owns (rows / tp_size)
    uint32_t owned_row_tiles = 0;  // owned_rows / TILE_HEIGHT
    uint32_t hidden_tiles = 0;     // hidden_dim / TILE_WIDTH
    uint32_t block_tiles = 0;      // a destination block's tile columns (hidden_tiles / tp_size)
    uint32_t worker_columns = 0;   // tile columns a worker (the largest divisor of block_tiles up to 8)
    uint32_t num_workers = 0;      // hidden_tiles / worker_columns
    uint32_t num_dram_banks = 0;
    uint32_t pages_per_run = 0;   // tiles a fabric packet (the fabric's max payload / a tile)
    uint32_t runs_per_group = 0;  // runs a port's staging group

    static TailAllToAllGeometry make(
        uint32_t tp_size, uint32_t rows, uint32_t hidden_dim, uint32_t num_dram_banks, size_t max_payload_bytes);
};

// Everything an all-to-all program needs that does not change between calls (built once per resources).
struct TailAllToAllPlan {
    TailAllToAllGeometry geometry;
    MeshTopology topology;  // the block's (SP rows x TP columns; transposed: SP on mesh axis 1)
    bool ring = false;
    uint32_t num_links = 0;
    TailSchedule schedule;
    // port core of (chip, direction, link); only directions with a neighbour
    std::map<std::tuple<ChipCoord, TailDirection, uint32_t>, tt::tt_metal::CoreCoord> ports;
    tt::tt_metal::CoreRangeSet port_union;  // every chip's port cores (the persistent words live there)
    std::vector<tt::tt_metal::CoreCoord> workers;
    std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> links;  // fabric link ids to each neighbour
};

std::shared_ptr<const TailAllToAllPlan> make_tail_all_to_all_plan(
    MoEBlockResources& resources, uint32_t rows, uint32_t hidden_dim);

// ---- latent engine ----
inline constexpr uint32_t kExchangeTiles = 4;  // a step's exchange: up to 3 partial tiles + the half's sum of squares
inline constexpr uint32_t kMaxArenaBytes = 1400 * 1024;
inline constexpr uint32_t kLandedGroupColumns = 3;  // one fused landed increment a 3 columns' shares
inline constexpr uint32_t kPacketsPerGroup = 8;     // packets a gather port's staging group

// The engine's geometry: rows of a mesh row, latent_dim, output_dim = H / TP, num_chunks = TP, an
// engine_columns x engine_rows engine.
struct TailEngineGeometry {
    uint32_t rows = 0, latent_dim = 0, output_dim = 0, num_chunks = 0, engine_columns = 0, engine_rows = 0;
    uint32_t w_tile_bytes = 0;               // W' tiles: bfloat8_b or bfloat16
    uint32_t latent_tiles = 0;               // latent_dim / 32
    uint32_t output_tiles = 0;               // output_dim / 32
    uint32_t row_tiles_per_chunk = 0;        // rows / 32 / num_chunks
    uint32_t num_pairs = 0;                  // engine_rows / 2: a pair = two cores, one a K half
    uint32_t max_steps = 0;                  // the most row tiles (steps) a pair has
    uint32_t max_own_steps = 0;              // the most of them from this chip's own rows
    uint32_t k_half_tiles = 0;               // latent_tiles / 2
    uint32_t max_column_tiles = 0;           // the most output tiles an engine column has (<= 7: DEST)
    uint32_t finished_ring_pages = 0;        // the finished columns' FIFOs: a multiple of every core's finished count
    uint32_t max_share_tiles = 0;            // the most K tiles a column's share has
    uint32_t ring_slots = 0;                 // the z ring's and the squares' slots
    std::vector<uint32_t> steps, own_steps;  // per pair
    std::vector<uint32_t> column_tiles, column_first_tile, finished_counts;
    std::vector<std::pair<uint32_t, uint32_t>> k_halves, w_k_ranges, share_k_ranges;
    // per column: (offset, count) of the columns half 0 / 1 finishes
    std::vector<std::array<std::pair<uint32_t, uint32_t>, 2>> finished_columns;
    uint32_t arena_bytes = 0;

    // capacity_bytes: the L1 the engine's CBs may take; ring_slots is the most that fit (>= min(2, max_steps)).
    static TailEngineGeometry make(
        uint32_t rows,
        uint32_t latent_dim,
        uint32_t output_dim,
        uint32_t num_chunks,
        uint32_t engine_columns,
        uint32_t engine_rows,
        uint32_t w_tile_bytes,
        uint32_t capacity_bytes);
    // (region offsets, total), 64 B aligned.
    std::pair<std::vector<uint32_t>, uint32_t> layout(uint32_t ring_slots) const;
    // Tiles in one W' region (a half column's K range, padded to the largest).
    uint32_t region_tiles() const;
};

// The watchers of the chip at `position`: {(column, row): [step | first column << 8 | end column << 16]} (sorted).
std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> engine_watchers(
    uint32_t position,
    uint32_t tp_size,
    bool ring,
    const TailEngineGeometry& geometry,
    uint32_t group_columns = kLandedGroupColumns);

struct TailEnginePlan {
    TailEngineGeometry geometry;
    MeshTopology topology;  // the block's (SP rows x TP columns; transposed: SP on mesh axis 1)
    bool ring = false;
    uint32_t num_links = 0;
    uint32_t num_dram_banks = 0;    // DRAM banks the W' regions are spread over (dram_grid_size().x)
    uint32_t tiles_per_packet = 0;  // tiles a gather packet
    float norm_eps = 0;             // the latent rmsnorm's eps
    std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> links;  // fabric link ids to each neighbour
    // the four gather ports (direction, link), at fixed cells in the column right of the engine (every chip)
    std::map<std::pair<TailDirection, uint32_t>, tt::tt_metal::CoreCoord> ports;
    tt::tt_metal::CoreRangeSet engine_cores;
    tt::tt_metal::CoreRangeSet port_cores;
};

// capacity_bytes: L1 the engine may use (at most kMaxArenaBytes).
std::shared_ptr<const TailEnginePlan> make_tail_engine_plan(
    MoEBlockResources& resources,
    uint32_t rows,
    uint32_t latent_dim,
    uint32_t output_dim,
    uint32_t w_tile_bytes,
    uint32_t capacity_bytes);

// The fabric link ids toward each row neighbour (MeshPlan's X+ / X- ports).
std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> row_links(const MeshPlan& plan);

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
