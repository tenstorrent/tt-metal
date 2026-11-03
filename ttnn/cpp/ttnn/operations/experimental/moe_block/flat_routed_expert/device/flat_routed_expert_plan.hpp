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
// the same plan). One subgrid; DRISC-only gate/up weight readers (the DRAM cores send every slot's weights, one Tensix
// launcher posts the jobs); even gate/up and down splits; x read once by the helper relays and fed along unicast
// snakes.
struct FlatRoutedExpertPlan {
    using Core = tt::tt_metal::CoreCoord;
    // shapes
    uint32_t H = 0, I = 0, E = 0, NG = 0, m = 0, Ht = 0, It = 0;
    uint32_t w_tile = 0;  // weight tile bytes
    uint32_t banks = 0;
    // gate/up split, sub-blocks
    uint32_t n_rd_sg = 16;
    uint32_t np = 1, g = 1, mt = 1, mtg = 1, x_slots = 24, dst_tiles = 8;
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
    // the program's static (non-arena) CBs, bytes a core at most: meta (6), counts / regions / ids (7), the schedule
    // inputs (9), gate/up CB 16 and h_local (3). A caller sizing the arena leaves this much L1 below it (the program
    // factory checks its CBs against it)
    uint32_t static_l1_bytes = 0;
    // the one Tensix core that posts every slot's DRISC job (the other reader cells are free: the down cores take them)
    Core launcher{0, 0};
    // weight slots served by the DRAM cores (one per reader cell; two DRISC tiles per bank in switch mode, or one per
    // bank serving both of its slots: drisc_tiles)
    uint32_t drisc_n = 0;
    FlatDriscTiles drisc_tiles = FlatDriscTiles::both;
    // DMA reads in flight per DRISC job (se_drisc.hpp SD_DMA_CAP; 0: up to its L1 slots)
    uint32_t drisc_dma_cap = 2;
    // blocks per DRISC unit (se_drisc.hpp SD_UB; 0: the sender's choice)
    uint32_t drisc_ub = 0;
    uint32_t drisc_mpfe = 0;       // (SD_MPFE: 0 untouched)
    bool drisc_early_free = true;  // (SD_UB_EARLY_FREE: the senders apply it to units of 2+ blocks)
    FlatDriscProbe drisc_probe = FlatDriscProbe::none;
    // who reads the gate/up weights (layout.gu_source) and, for the gate/up cores' own reads, how many in flight
    FlatGuSource gu_source = FlatGuSource::drisc;
    uint32_t gu_depth = 2;
    uint32_t gu_pieces = 1;
    // per gate/up core (p.gu order): who delivers its weights (0 its slot's DRISC, FlatGuPlace::tensix_noc0 /
    // tensix_noc1 its own reads); per weight slot: served by a DRISC
    std::vector<uint32_t> gu_how;
    std::vector<bool> slot_drisc;
    // the expected rows per expert the plan was made for (layout.rows_hint; -1 unknown): the placement's x weight
    int32_t rows_hint = -1;
    // the relays' copy of the call's token index (indexed x): 4 B per capacity row at ti_off, read once per call
    uint32_t ti_off = 0;
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

    std::vector<Core> arena_cores() const;  // gu + down + relays + gu_idle + launcher
    uint32_t rect_of(const Core& c) const;
    static uint32_t kd_of(uint32_t pcd, uint32_t it);
};

struct FlatGuGeom;
struct FlatGuAssign;
struct FlatGuTraffic;
// the link-load placement search's iterations (plan time, per distinct chip geometry)
inline constexpr uint32_t kFlatGuIters = 20000;
// Sets the plan's gate/up cores (slot order), their delivery and the DRISC slots from FlatLayoutOptions.gu_place
// words (one per gate/up cell)
void flat_apply_gu_place(FlatRoutedExpertPlan& p, const std::vector<uint32_t>& words);
// The auto sources' placement for one chip's geometry, and the traffic the search weighs
FlatGuAssign flat_gu_auto(const FlatRoutedExpertPlan& p, const FlatGuGeom& g);
FlatGuTraffic flat_gu_traffic(const FlatRoutedExpertPlan& p);

// Builds the plan for the mesh device's chips (all of them laid out as its first local chip, as the program is).
FlatRoutedExpertPlan make_flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& cfg);

// The plan for a mesh device and config, cached for the process (keyed by the mesh device's id and shape and the
// config fields the plan depends on): what the program factory, the weight layout and the done words use.
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& cfg);

// Chip `coord`'s plan: the mesh plan, with the gate/up placement of that chip's own geometry for the auto sources
// (gu_source auto_tensix / auto_hybrid; harvesting differs per chip); else the mesh plan itself. Cached per distinct
// geometry. Only the gate/up cores' order, delivery and DRISC slots differ from the mesh plan (same cells, same
// weight layout).
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_chip_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertConfig& cfg,
    const tt::tt_metal::distributed::MeshCoordinate& coord);

// Greedy maximal rectangles over a core list (one range per core makes every dispatch write a unicast).
tt::tt_metal::CoreRangeSet rect_ranges(const std::vector<tt::tt_metal::CoreCoord>& cores);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
