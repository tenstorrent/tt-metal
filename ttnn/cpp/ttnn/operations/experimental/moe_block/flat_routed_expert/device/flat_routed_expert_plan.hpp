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
    // shapes
    uint32_t H = 0, I = 0, E = 0, NG = 0, m = 0, Ht = 0, It = 0;
    uint32_t w_tile = 0;  // weight tile bytes
    uint32_t banks = 0;
    // gate/up split, sub-blocks
    uint32_t n_rd_sg = 16;
    uint32_t np = 1, mt = 1, x_slots = 24, dst_tiles = 8;
    bool gu_rp = false;
    uint32_t s = 1, v = 1, nk_gu = 1, slot = 1, rg = 1, r_ = 1, ring_g = 1;
    // half-width gate / up tiles (16-column units): a unit's weight tile holds [gate | up] of 16 columns (the
    // activation runs face-wise, se_glu_faces.hpp)
    bool hg = false;
    // 16-column units (hg; np is then 0): a core's gate/up block is nu weight tiles a row ([gate | up] of units cg nu
    // ..), its h a run of hpc h tiles from K tile cg nu / 2, the first / last one half its own when cg nu / (cg + 1) nu
    // is odd. Generic widths: gwt weight tiles a row (2 np / nu), hpc h tiles a row (np / (nu + 1) / 2), ngrp column
    // groups (It / np / 2 It / nu)
    uint32_t nu = 0, gwt = 2, hpc = 1, ngrp = 0;
    // gate/up core j of weight slot r
    uint32_t gu_at(uint32_t r, uint32_t j) const { return r * r_ + j; }
    // relays
    uint32_t nh = 1, land_slots = 3, vstride = 2, sbt = 32, seg = 2048, nsb = 1;
    // gate/up fp32 in a full-sync DST (the whole sub-block in one K loop: wide gate/up blocks, gwt > 4)
    bool gu_full_sync = false;
    // gate/up K blocks per CB handshake in the row-pass K loop (1 without row passes)
    uint32_t gu_kg = 1;
    // the program's static (non-arena) CBs, bytes a core at most: meta (6), counts / regions / ids (7), gate/up CB 16
    // and h_local (3). A caller sizing the arena leaves this much L1 below it (the program
    // factory checks its CBs against it)
    uint32_t static_l1_bytes = 0;
    // reader cell 0: idle (kept out of the relay / down pool, so the cells the others take are where they were)
    Core spare{0, 0};
    // weight slots (one per reader cell, slot r in bank r % banks)
    uint32_t n_slots = 0;
    // weight reads in flight a gate/up core (pieces), and the pieces a weight block is read as (layout.gu_read_pieces)
    static constexpr uint32_t gu_depth = 2;
    uint32_t gu_pieces = 1;
    // down
    uint32_t d_ch = 7, rem_cols = 0, h_tiles = 0, hbuf = 3;
    float dring = 2.0f;
    std::vector<uint32_t> pcds, col0s;
    // arena (per core, 2 KB aligned offsets)
    uint32_t x_off = 0, hl_off = 0, h_off = 0, o_off = 0, sb_off = 0, land_off = 0, arena_tiles = 0;
    // arena bytes each role uses (the arena is max of them: the rest of a role core's arena is free)
    uint32_t gu_bytes = 0, dn_bytes = 0, relay_bytes = 0;
    // weight regions (bytes per core region)
    uint32_t region_bytes = 0;
    // the down cores' weight blocks rotate over the banks: block k of down core d in bank (d + k) % banks, at
    // d * wd_rot_region + (k / banks) blocks, like unified_routed_expert_ffn's one-tile-row ND shards
    uint32_t wd_rot_region = 0;
    // role cores (logical)
    std::vector<Core> readers, gu, gu_idle, relays, down, coords;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> rects;  // x0, x1, y0, y1
    std::map<uint32_t, uint32_t> d_pred, d_succ;
    std::vector<uint32_t> d_heads;

    std::vector<Core> arena_cores() const;  // gu + down + relays + gu_idle + spare
    uint32_t rect_of(const Core& c) const;
    static uint32_t kd_of(uint32_t pcd, uint32_t it);
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
