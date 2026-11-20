// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "flat_routed_expert_device_operation_types.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// Layout constants shared by the planner and the program factory.
inline constexpr uint32_t kKBlk = 8;        // K tiles per gate/up weight / x block
inline constexpr uint32_t kBf8Tile = 1088;  // bfp8 tile bytes
inline constexpr uint32_t kRmChunks = 4;    // x relay: row-major chunks (32 rows x SBT tiles) buffered
inline constexpr uint32_t kSbSlots = 3;     // x relay: tilized super-blocks buffered

// The spatial pipeline's layout: role cores, the work split and the per-core L1 arena. A pure function of the
// device grid and the config (the program factory, the host wrapper's scratch tensors and the weight layout all use
// the same plan). One subgrid; the gate/up cores read their own weight blocks (each weight slot's cores from the
// slot's bank); even gate/up and down splits; x read once by the helper relays and fed along unicast snakes.
struct FlatRoutedExpertPlan {
    using Core = tt::tt_metal::CoreCoord;

    // ---- shapes
    uint32_t hidden = 0, intermediate = 0;              // H, I (elements)
    uint32_t experts_per_chip = 0, global_experts = 0;  // E, the global expert count
    uint32_t max_tokens_per_expert = 0;                 // m: the capacity
    uint32_t hidden_tiles = 0, intermediate_tiles = 0;  // H / 32, I / 32
    uint32_t weight_tile_bytes = 0;                     // a weight tile (bfp8 1088, bfp4 576)
    uint32_t x_tile_bytes = kBf8Tile;                   // an x tile after the tilize (bf16 2048 with layout.x_bf16)
    uint32_t h_tile_bytes = kBf8Tile;                   // an h tile after gate/up (bf16 2048 with layout.h_bf16)
    uint32_t dram_banks = 0;

    // ---- the gate/up split and its sub-blocks
    // weight slots the gate/up cores' weights are packed in: one a DRAM bank (slot r in bank r), the gate/up cores of a
    // slot reading its region
    uint32_t gate_up_weight_slots = 8;
    // whole gate / up tile pairs a gate/up core (0 with 16-column units: units_per_core)
    uint32_t gate_up_pairs_per_core = 1;
    uint32_t row_tiles_per_sub_block = 1;  // row tiles (32 rows) a gate/up sub-block
    uint32_t x_ring_blocks = 24;           // the gate/up cores' x ring (K-blocks)
    uint32_t gate_up_dst_tiles = 8;        // DST tiles a gate/up sub-block may use
    bool gate_up_row_passes = false;     // gate/up runs a sub-block's K-blocks once a row tile (the x ring holds them)
    uint32_t sub_blocks_per_expert = 1;  // the capacity's sub-blocks (m rounded up to whole sub-blocks)
    uint32_t sub_blocks_per_chip = 1;    // experts_per_chip x sub_blocks_per_expert
    uint32_t gate_up_k_blocks = 1;       // K-blocks (kKBlk tiles) over H
    uint32_t weight_block_tiles = 1;     // a gate/up weight block: kKBlk x gate_up_weight_tiles_per_row
    uint32_t column_groups_per_weight_slot = 1;  // gate/up column groups a weight slot holds
    uint32_t gate_up_cores_per_weight_slot = 1;  // gate/up cores reading a weight slot (as many as its column groups)
    uint32_t gate_up_weight_ring_blocks = 1;     // the gate/up weight ring: 2 expert slices (K-blocks)
    // half-width gate / up tiles (16-column units): a unit's weight tile holds [gate | up] of 16 columns (the
    // activation runs face-wise, glu_faces.hpp)
    bool half_width_gate_up = false;
    // 16-column units (half_width_gate_up; gate_up_pairs_per_core is then 0): a core's gate/up block is
    // units_per_core weight tiles a row ([gate | up] of units cg units_per_core ..), its h a run of h_tiles_per_core h
    // tiles from K tile cg units_per_core / 2, the first / last one half its own when cg units_per_core /
    // (cg + 1) units_per_core is odd. Generic widths: gate_up_weight_tiles_per_row (2 pairs / units),
    // h_tiles_per_core (pairs / (units + 1) / 2), column_groups (It / pairs / 2 It / units)
    uint32_t units_per_core = 0, gate_up_weight_tiles_per_row = 2, h_tiles_per_core = 1, column_groups = 0;
    // gate/up core j of weight slot r
    uint32_t gate_up_core_index(uint32_t r, uint32_t j) const { return r * gate_up_cores_per_weight_slot + j; }

    // ---- the x relays
    uint32_t helper_relays_per_primary = 1;
    uint32_t relay_landing_slots = 3;   // super-block slots in a primary relay's landing rings
    uint32_t relay_stride = 2;          // super-blocks rotate over the relays: 1 + helper_relays_per_primary
    uint32_t super_block_tiles = 32;    // x tiles (K) a super-block
    uint32_t row_segment_bytes = 2048;  // a row-major row's slice of a super-block (super_block_tiles x 64 B)
    uint32_t super_blocks = 1;          // super-blocks over H
    // gate/up fp32 in a full-sync DST (the whole sub-block in one K loop: wide gate/up blocks, more than 4 tiles a row)
    bool gate_up_full_sync = false;
    // the program's static (non-arena) CBs, bytes a core: 0 - every CB is in the arena (h_local 3 and the pack CB 16 on
    // the gate/up cores, meta 6 and counts / regions / ids 7 at one offset on every arena core). Kept for callers that
    // size the L1 below the arena (nothing to leave).
    uint32_t static_l1_bytes = 0;
    // the schedule page (CB 6) and the dynamic counts (CB 7): bytes, and 4 x the schedule scratch buffer the program
    // factory sizes (schedule_scratch_bytes)
    uint32_t schedule_page_bytes = 0, dynamic_counts_bytes = 0;
    // weight reads in flight a gate/up core (pieces), and the pieces a weight block is read as (the most of at least
    // 6 KB each)
    static constexpr uint32_t gate_up_reads_in_flight = 2;
    uint32_t gate_up_block_pieces = 1;

    // ---- the down projection
    uint32_t down_chains = 7;               // the chains the gate/up cores' h is forwarded along
    uint32_t down_output_columns = 0;       // output tile columns over the down cores (hidden_tiles)
    uint32_t down_h_buffer_tiles = 0;       // one down h buffer: a sub-block's h
    uint32_t down_h_buffers = 3;            // h buffers a down core
    float down_weight_ring_experts = 2.0f;  // the down weight ring: experts it holds (2 pinned, else 1.5)
    std::vector<uint32_t> down_columns_per_core, down_first_columns;  // per down core: its output columns, first one

    // ---- the arena (per core, 2 KB aligned offsets)
    // the gate/up cores: weight ring at 0, x ring, h_local (CB 3), the pack CB (16); the down cores: weight ring at 0,
    // h buffers, output; the helper relays: row-major chunks at 0, super-blocks, landing rings; every arena core: the
    // schedule page (CB 6) and the dynamic counts (CB 7) at one offset past every role's own bytes (one address on all
    // of them, as the static CBs had)
    uint32_t x_ring_offset = 0, h_local_offset = 0, pack_cb_offset = 0;
    uint32_t down_h_offset = 0, down_output_offset = 0;
    uint32_t super_block_offset = 0, relay_landing_offset = 0;
    uint32_t arena_tiles = 0;
    uint32_t schedule_page_offset = 0, dynamic_counts_offset = 0;
    // arena bytes each role uses (the arena is max of them: the rest of a role core's arena is free)
    uint32_t gate_up_arena_bytes = 0, down_arena_bytes = 0, relay_arena_bytes = 0;
    // a weight slot's gate/up region (bytes per core region)
    uint32_t gate_up_weight_region_bytes = 0;
    // the down cores' weight blocks rotate over the banks: block k of down core d in bank (d + k) % dram_banks, at
    // d * down_weight_region_bytes + (k / dram_banks) blocks, like unified_routed_expert_ffn's one-tile-row ND shards
    uint32_t down_weight_region_bytes = 0;

    // ---- role cores (logical)
    // each DRAM bank's anchor: its optimal worker core (gate/up cores are placed by their NOC0 hops from it)
    std::vector<Core> bank_anchor_cores;
    std::vector<Core> gate_up_cores, idle_gate_up_cores, relay_cores, down_cores;
    std::vector<Core> coordinator_cores;  // the core that holds the call's done words (the first down core)
    // the gate/up rectangles: x0, x1, y0, y1
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> gate_up_rectangles;
    // the down chains: each down core's predecessor / successor, and the chains' first cores
    std::map<uint32_t, uint32_t> down_chain_predecessor, down_chain_successor;
    std::vector<uint32_t> down_chain_heads;

    // gate_up_cores + down_cores + relay_cores + idle_gate_up_cores
    std::vector<Core> arena_cores() const;
    // The arena a call leaves free on the flat expert cores, up to arena_end (the caller's arena bytes a core): cores
    // with the same free spans ([begin, end) offsets, 64 B aligned) - what the combine lands rows in
    // (CombineLanding::FlatCores). Gate/up: past its own bytes up to CBs 6 / 7; primary relays: below their landing
    // rings (they read / tilize nothing) and past them; helper relays: from the landing rings' offset (they only send);
    // idle / spare cores: all of it (no kernel there, no CB bound); every arena core: past CBs 6 / 7. Down cores: past
    // CBs 6 / 7 (their out CB takes the room up to them).
    struct FreeL1 {
        std::vector<Core> cores;
        std::vector<std::pair<uint32_t, uint32_t>> spans;
    };
    std::vector<FreeL1> free_l1(uint32_t arena_end) const;
    // the cores that run kernels / bind CBs 6 / 7 (gate/up, down, relays)
    std::vector<Core> work_cores() const;
    // the gate/up rectangle a core is in
    uint32_t rectangle_of(const Core& c) const;
    // a down core's K-block (tiles) at its output columns
    static uint32_t down_k_block_tiles(uint32_t columns, uint32_t intermediate_tiles);
};

// Builds the plan for the mesh device's chips (all of them laid out as its first local chip, as the program is).
FlatRoutedExpertPlan make_flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& cfg);

// The plan for a mesh device and config, cached for the process (keyed by the mesh device's id and shape and the
// config fields the plan depends on): what the program factory, the weight layout and the done words use.
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& cfg);

// Greedy maximal rectangles over a core list (one range per core makes every dispatch write a unicast).
tt::tt_metal::CoreRangeSet rect_ranges(const std::vector<tt::tt_metal::CoreCoord>& cores);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
