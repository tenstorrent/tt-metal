// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's host sizing, no device: the pieces head_plan_fit needs to decide whether a layout builds.
//  * ReduceScatterPlan: the SE down + reduce-scatter's tiling and transport schedule (over plan_transport).
//  * SharedExpertSizing: the matmul cores' arena plan.
//  * router_layout / router_min_bytes: the router cores' arena regions.
//  * routing_exchange_sizes / worst_rows / slot_bytes: the routing exchange's buffers.
//  * routing_paths: the routing exchange's senders.
// A requirement a layout does not meet is a SizingError saying which one, with its numbers.

#pragma once

#include <algorithm>
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

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace head_math {
inline int64_t align_to_64(int64_t value) { return (value + 63) / 64 * 64; }
inline int64_t align_to_16(int64_t value) { return (value + 15) / 16 * 16; }
inline int64_t ceil_div(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }
// value mod modulus in [0, modulus)
inline int64_t positive_modulo(int64_t value, int64_t modulus) { return ((value % modulus) + modulus) % modulus; }
}  // namespace head_math

// A requirement of the build failed (what(): which one, with its numbers).
struct SizingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The matmul cores' arena does not fit its cap at any W ring / h slot count.
struct ArenaTooLarge : SizingError {
    ArenaTooLarge(const std::string& what, int64_t need, int64_t cap) : SizingError(what), need(need), cap(cap) {}
    int64_t need;
    int64_t cap;
};

// Ordered (name, bytes) regions, in arena order.
using Regions = std::vector<std::pair<std::string, int64_t>>;

inline constexpr int64_t kArenaMax = 1344 * 1024;  // the arena's largest size a core
// The kernels' args blocks (head_workload.cpp head_args_to_dram): a region past the arena's regions on every arena core
// - the RISCV_0 kernel's block, the RISCV_1 kernel's, the compute kernel's page. Taken off the arena's cap.
inline constexpr int64_t kHeadArgsBytes = 12 * 1024;
inline constexpr uint32_t kDestTiles = 8;  // the compute kernel's DEST tiles

// port_cb_bytes: a port core's packet CB at K step k_step_tiles (an arena region on the port cores).
int64_t port_cb_bytes(uint32_t k_step_tiles, uint32_t payload, uint32_t group = 4);
// arena_cap_of: L1 the arena may take from a free-L1 figure, above cb_floor bytes of other static CBs (every
// head CB is in the arena; the MoE block's programs that run in it after the head have none either: 0 there) and below
// the kernels' args region (kHeadArgsBytes).
int64_t arena_cap_of(int64_t l1_free, int64_t cb_floor = 0);

// `total` into `parts` widths, multiples of unit, as even as possible (the first total / unit % parts one unit
// wider).
std::vector<uint32_t> column_split(uint32_t total, uint32_t parts, uint32_t unit);

// ---- the reduce-scatter's transport schedule (mirror off, one wave) ----
enum class ReduceScatterKind : uint8_t { Line, RingBalanced, RingUnbalanced };
enum class ReduceScatterDirection : uint8_t { Forward = 0, Backward = 1, Final = 2 };
ReduceScatterKind reduce_scatter_kind(uint32_t tp_size, bool ring);

struct ReduceScatterRole {
    ReduceScatterDirection direction;
    bool start;
    uint32_t distance;
};

struct ReduceScatterChip {
    std::map<std::pair<uint32_t, uint32_t>, ReduceScatterRole> roles;  // (block, half)
    std::vector<uint32_t> entries[2];                                  // per half
    std::map<std::pair<uint32_t, ReduceScatterDirection>, std::vector<uint32_t>>
        arrivals;  // (half, forward | backward)
};

struct ReduceScatterTransport {
    std::vector<ReduceScatterChip> chips;
    uint32_t slots[2] = {1, 1};  // forward, backward
};
ReduceScatterTransport plan_transport(uint32_t tp_size, ReduceScatterKind kind);

// ---- ReduceScatterPlan (the parts the sizing reads) ----
class ReduceScatterPlan {
public:
    struct Pass {
        uint32_t first_column;
        uint32_t width;
        std::optional<uint32_t> entry;
    };
    enum class Operation : uint8_t { Pass, Handoff, Final };

    ReduceScatterPlan(
        uint32_t tp_size,
        bool ring,
        uint32_t engine_columns,
        uint32_t engine_rows,
        uint32_t row_tiles_per_core,
        uint32_t hidden_tiles,
        int scatter,
        uint32_t links,
        uint32_t max_pass_width,
        bool compact_reduce_scatter_zones);

    uint32_t tp_size, engine_columns, engine_rows, row_tiles_per_core, hidden_tiles, links, max_pass_width,
        chunk_row_tiles, num_cores;
    int scatter;
    bool compact_reduce_scatter_zones;
    ReduceScatterKind kind;
    ReduceScatterTransport transport;
    uint32_t core_block_tiles = 0;
    // scatter -1
    uint32_t block_columns = 0, passes_per_block = 0;
    std::vector<uint32_t> block_column_widths, block_column_starts;
    // scatter -2
    uint32_t block_row_tiles = 0, num_passes = 0;
    std::vector<uint32_t> column_widths, column_starts;
    std::optional<uint32_t> half_split_column;
    bool pass_major = false;

    uint32_t core_index(uint32_t core_x, uint32_t core_y) const { return core_y * engine_columns + core_x; }
    uint32_t half(uint32_t core) const;
    std::pair<uint32_t, uint32_t> rows_in(uint32_t core_y, uint32_t block) const;
    uint32_t count(uint32_t core_x, uint32_t core_y, uint32_t block) const;
    const std::vector<uint32_t>& entries(uint32_t position, uint32_t half_index = 0) const {
        return transport.chips[position].entries[half_index];
    }
    const ReduceScatterRole& role(uint32_t position, uint32_t block, uint32_t half_index) const;
    std::vector<Pass> column_passes(uint32_t position, uint32_t core_x) const;
    uint32_t final_block_tiles(uint32_t core_x, uint32_t core_y) const;
    // (offsets by entry, HOLD tiles used)
    std::pair<std::map<uint32_t, uint32_t>, uint32_t> hold_offsets(
        uint32_t position, uint32_t core_x, uint32_t core_y) const;
    std::vector<std::pair<Operation, uint32_t>> operations(uint32_t position, uint32_t core_x, uint32_t core_y) const;
    std::vector<std::pair<uint32_t, ReduceScatterDirection>> handoffs(
        uint32_t position, uint32_t core_x, uint32_t core_y) const;
    std::optional<uint32_t> down(uint32_t position, ReduceScatterDirection direction) const;
    std::optional<uint32_t> piece_entry(uint32_t position, uint32_t half_index, ReduceScatterDirection direction) const;
    std::vector<std::pair<uint32_t, uint32_t>> zone_slots(
        uint32_t position, uint32_t core_x, uint32_t core_y, ReduceScatterDirection direction) const;
    uint32_t zone_tiles(ReduceScatterDirection direction) const;

    // ---- the device build's tables ----
    struct Arrival {
        uint32_t entry, slot, threshold;
    };
    struct PortUnit {
        uint32_t entry, block, mask, slot0, slot1;
        bool pieces;
    };
    uint32_t link(uint32_t core) const { return core % links; }
    std::optional<uint32_t> up(uint32_t position, ReduceScatterDirection direction) const;
    // per direction (forward, backward): [(entry, zone slot, threshold)]
    std::array<std::vector<Arrival>, 2> arrivals(uint32_t position, uint32_t core_x, uint32_t core_y) const;
    std::vector<PortUnit> port_units(uint32_t position, ReduceScatterDirection direction) const;
    std::vector<ReduceScatterDirection> final_block_readers(uint32_t position, uint32_t core_x, uint32_t core_y) const;
    // (core_x, core_y) of link link_index, in core index order
    std::vector<std::pair<uint32_t, uint32_t>> port_cores(uint32_t link_index) const;
    std::tuple<uint32_t, uint32_t, uint32_t> final_map(uint32_t position, uint32_t core_x, uint32_t core_y) const;
    uint32_t prefix_mask(uint32_t position, uint32_t half_index, ReduceScatterDirection direction, uint32_t slot) const;
    uint32_t counts_word(uint32_t core_x, uint32_t core_y) const;
    uint32_t final_block_offset(uint32_t position, uint32_t core_x, uint32_t core_y, uint32_t entry) const;
    std::vector<uint32_t> pass_widths(uint32_t core_x) const;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> pass_bases(
        uint32_t position, uint32_t core_x, uint32_t core_y) const;
    std::vector<uint32_t> compute_args(uint32_t position, uint32_t core_x, uint32_t core_y) const;

private:
    bool needs_hold(uint32_t position, uint32_t core_x, uint32_t core_y, uint32_t entry) const;
};

// ---- the shared expert's (matmul cores') host sizing ----
enum class WeightDtype : uint8_t { Bf4, Bf8, Bf16 };

struct SharedExpertArgs {
    uint32_t rows = 0, hidden_dim = 0, shared_expert_dim = 0;
    uint32_t engine_columns = 11, engine_rows = 10, row_tiles_per_core = 1, k_step_tiles = 16;
    uint32_t shards = 1;
    const ReduceScatterPlan* reduce_scatter = nullptr;  // the down + reduce-scatter's plan (required)
    std::optional<uint32_t> latent;                     // tiles a chip
    std::pair<uint32_t, uint32_t> router{};  // (expert tiles, router K tiles): the router logits, on the engine
    uint32_t router_iterations = 1;  // the router stage over row_chunks / router_iterations row chunks at a time
    int64_t arena_max = kArenaMax;
    WeightDtype w_dtype = WeightDtype::Bf8;
};

struct SharedExpertSizing {
    explicit SharedExpertSizing(const SharedExpertArgs& sizing_args);  // throws SizingError / ArenaTooLarge

    SharedExpertArgs args;  // (args.reduce_scatter: not owned - valid while the caller's ReduceScatterPlan is)
    uint32_t engine_columns, engine_rows, row_tiles_per_core, chunk_row_tiles, rows, hidden_dim, shared_expert_dim,
        shards;
    uint32_t row_chunks, hidden_tiles, shared_expert_tiles, k_step_tiles, num_k_steps, k_steps_per_shard, family0_size,
        family1_size, gate_up_passes, gate_up_passes_without_latent, down_passes, down_k_step_tiles, down_k_steps,
        w_blocks_per_chunk, w_slot_tiles, w_tile_bytes;
    uint32_t max_column_pairs, max_down_pass_width, gate_up_pass_width,
        max_latent_column_tiles = 0, latent_tiles = 0, w_ring_slots = 0, h_slots = 1, prep_full_column_tiles = 0,
        prep_block_column_tiles = 0;
    int64_t latent_pages = 0, x_own_tiles = 0, w_own_tiles = 0, accumulator_pages = 0;
    std::vector<uint32_t> family0_columns, family1_columns;  // the x families' owner columns
    std::vector<uint32_t> column_pairs, column_first_pair, latent_column_tiles, latent_column_first_tile;
    std::vector<std::vector<uint32_t>> gate_up_pass_widths, down_pass_widths;
    // the router logits on the engine (args.router)
    uint32_t router_expert_tiles_per_column = 0, router_row_tiles_per_group = 0, router_k_chunk_tiles = 0;
    bool router_overlays_zones = false, router_overlays_latent = false, router_overlay = false;
    Regions reduce_scatter_sizes;  // the reduce-scatter's regions
    int64_t arena_bytes = 0;
    Regions offsets;                       // region offsets, arena order
    Regions sizes;                         // each region's bytes (overlays share their host's)
    std::vector<int64_t> sequential_ends;  // the sequential regions' bounds (the CB ring check's)
    // compute kernel groups: columns with the same (gate / up widths, down widths), in first-column order
    std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>, std::vector<uint32_t>>> groups;

    int64_t dram_w_bytes() const {
        return int64_t(hidden_tiles * 2 * shared_expert_tiles + shared_expert_tiles * hidden_tiles) * w_tile_bytes;
    }
    int64_t at(const std::string& region_name) const;  // offsets[region_name]
    bool in_family0(uint32_t column) const;
    bool in_family1(uint32_t column) const;
    // (owner column, family, index in the family) of x step `step`
    std::tuple<uint32_t, uint32_t, uint32_t> owner(uint32_t step) const;
    // chunk `chunk`'s local row tile local_row_tile -> the global row tile
    uint32_t global_row_tile(uint32_t chunk, uint32_t local_row_tile) const;
    // (first local row, rows) of core row core_y's row_tiles_per_core row tiles in owner block `block` (latent / prep)
    std::pair<uint32_t, uint32_t> latent_rows(uint32_t core_y, uint32_t block) const;
    uint32_t w_block_tiles(uint32_t column, uint32_t block) const;
    int64_t latent_ready_bytes() const { return 16 * int64_t(std::max<uint32_t>(4, row_chunks)); }
    bool final_rows() const;
};

// ---- the router cores ----
// (offsets, bytes) of the router cores' arena regions.
// arrivals_first (row blocks): the peers' logits arrivals before the bias, so the last block's routing planner overlays
// them too (they are dead once every router core's rows are written; an earlier block's planner never overlays them - a
// peer one block ahead writes its next logits there).
// full_expert_tiles: the bias's expert tiles (0: expert_tiles; else a "part" region too).
std::pair<Regions, int64_t> router_layout(
    uint32_t tp_size,
    uint32_t expert_tiles,
    uint32_t top_k,
    uint32_t full_expert_tiles = 0,
    bool arrivals_first = false);
int64_t router_min_bytes(uint32_t tp_size, uint32_t experts, uint32_t top_k);

// ---- the routing exchange ----
int64_t routing_header_bytes(uint32_t experts_per_chip, uint32_t instances);
int64_t routing_slot_bytes(uint32_t tokens_per_chip, uint32_t top_k, uint32_t experts_per_chip, uint32_t instances);
int64_t routing_worst_rows(uint32_t num_chips, uint32_t tokens_per_chip, uint32_t top_k, uint32_t experts_per_chip);
struct RoutingSizes {
    uint32_t planner_instances;
    int64_t entries_capacity, planner_l1_bytes, port_l1_bytes;
    bool fits;
    std::string reason;
};
// instances: the planner's instances (0: min(8, experts_per_chip, accumulators)).
RoutingSizes routing_exchange_sizes(
    uint32_t num_chips,
    uint32_t tokens_per_chip,
    uint32_t top_k,
    uint32_t experts_per_chip,
    uint32_t accumulators,
    int64_t rows_capacity,
    uint32_t instances = 0,
    uint32_t tokens_per_page = 32);

// The routing planner's instances on the router cores (num_router_cores of them, one each - its BRISC, NOC1): the
// most instances (<= num_router_cores + num_planner_only_cores, experts_per_chip, accumulators, 14) whose regions
// (one a core, step bytes) fit - over the router's dead buffers [0, overlay) when they fit there, else past the
// router's bytes (router_bytes) - in the arena. fixed_instances: that many (an earlier row block's planner: the last's
// instances over the router's local buffers [0, arr) - never the peers' logits arrivals).
// num_planner_only_cores (row blocks): planner-only cores - their whole arena the planner's. The instances fill the
// planner-only cores first (instance < num_planner_only_cores: that core), then the router cores (instance -
// num_planner_only_cores); without planner-only cores instance i is router core i's.
// Every instance runs on its core's BRISC, on NOC1 - the fabric routers' NoC: the routing packets land in DRAM through
// the NOC1 endpoint and the router's rows are written on NOC1, so every planner reads them through the port they were
// written through (a read through the other NoC's endpoint can return the page's previous contents after the write's
// ack: the BH DRAM cross-port stale read). One NOC1 RISC a core (two RISCs on one NoC share its counters).
// planner_base(instance): its region's offset.
struct PlannerSlot {
    bool router;    // a router core's (else a planner-only core's)
    uint32_t core;  // its index among the router / the planner-only cores
    uint32_t risc;  // 0 BRISC, 1 NCRISC (its region: the core's risc-th)
};
inline PlannerSlot planner_slot(uint32_t instance, uint32_t num_router_cores, uint32_t num_planner_only_cores) {
    if (instance < num_planner_only_cores) {
        return {false, instance, 0};
    }
    instance -= num_planner_only_cores;
    // (place_planner: instances <= num_router_cores + num_planner_only_cores - one instance a core)
    return {true, instance % num_router_cores, 0};
}
struct PlannerPlace {
    bool ok = false;
    uint32_t planner_instances = 0;
    RoutingSizes sizes{};
    int64_t step = 0, router_region_start = 0;
    uint32_t num_router_cores = 0, num_planner_only_cores = 0;
    int64_t planner_base(uint32_t instance) const {
        const auto slot = planner_slot(instance, num_router_cores, num_planner_only_cores);
        return (slot.router ? router_region_start : 0) + int64_t(slot.risc) * step;
    }
};
PlannerPlace place_planner(
    uint32_t num_chips,
    uint32_t tokens_per_chip,
    uint32_t top_k,
    uint32_t experts_per_chip,
    uint32_t accumulators,
    uint32_t num_router_cores,
    int64_t overlay,
    int64_t router_bytes,
    int64_t arena,
    uint32_t fixed_instances = 0,
    uint32_t num_planner_only_cores = 0);
// row blocks: the planner cores a head wants (router cores and planner-only ones, two instances each)
inline constexpr uint32_t kPlannerCores = 14;
// The cells no head piece takes, (x, y): left of the mover column (the grid's last), past the matmul cores
// (engine_columns x engine_rows), less the router cores placed under the matmul cores (the first `under` of rows
// engine_rows .. by row, columns 0 .. engine_columns); the planner-only cores (row blocks) are the first of them.
inline std::vector<std::pair<uint32_t, uint32_t>> head_free_cells(
    uint32_t grid_x, uint32_t grid_y, uint32_t engine_columns, uint32_t engine_rows, uint32_t under) {
    std::vector<std::pair<uint32_t, uint32_t>> cells;
    uint32_t skipped = 0;
    for (uint32_t y = engine_rows; y < grid_y; ++y) {
        for (uint32_t x = 0; x < engine_columns; ++x) {
            if (skipped < under) {
                ++skipped;
                continue;
            }
            cells.emplace_back(x, y);
        }
    }
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = engine_columns; x + 1 < grid_x; ++x) {
            cells.emplace_back(x, y);
        }
    }
    return cells;
}
// row blocks: how many planner-only cores a head with num_router_cores router cores takes, of `free` free cells - up to
// every instance's (14 at most, one a core): with them all past the router cores, the router kernels carry no planner
// (one program running every block holds both planners - an earlier block's and the last's - and the router cores'
// would not fit the kernel config buffer)
inline uint32_t planner_only_cores(uint32_t row_blocks, size_t free) {
    if (row_blocks <= 1) {
        return 0;
    }
    return static_cast<uint32_t>(std::min<size_t>(free, kPlannerCores));
}
// an earlier row block's planner plans no token: sized for one row tile
inline constexpr uint32_t kPlannerIdleTokens = 32;
// ---- routing_paths ----
struct RoutingPaths {
    uint32_t max_x_packets;  // the most packets an X sender sends
    uint32_t max_y_packets;  // ... a Y sender
    uint32_t max_senders;    // the most senders a chip
};
// The routing exchange's paths on a mesh_rows x mesh_columns mesh, or why they do not build.
// first_hops: the fabric's first hop between every pair of chips (HeadShape.first_hops).
std::variant<RoutingPaths, std::string> routing_paths(
    uint32_t mesh_rows, uint32_t mesh_columns, bool ring, uint32_t links, const std::vector<uint8_t>& first_hops);

// "%.<precision>f" of value, and "[a, b, ...]" of a list (the planner's messages).
std::string format_fixed(double value, int precision);
std::string list_string(const std::vector<uint32_t>& values);

}  // namespace ttnn::operations::experimental::moe_block::moe_head
