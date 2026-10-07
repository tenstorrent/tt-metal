// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {
class IDevice;
}

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

// Layout constants shared by the planner and the program factory.
inline constexpr uint32_t kKBlk = 8;        // K tiles per gate/up weight / x block
inline constexpr uint32_t kBf8Tile = 1088;  // bfp8 tile bytes
inline constexpr uint32_t kRmChunks = 4;    // x relay: row-major chunks (32 rows x SBT tiles) buffered
inline constexpr uint32_t kSbSlots = 3;     // x relay: tilized super-blocks buffered

// What the flat routed expert is built for. Every field is part of the program-cache key.
struct FlatRoutedExpertConfig {
    uint32_t hidden = 0;              // H (x / y width)
    uint32_t intermediate = 0;        // I (per device: I / TP)
    uint32_t experts_per_chip = 0;    // E
    uint32_t num_global_experts = 0;  // NG: the counts / regions rows' length
    uint32_t max_tokens = 0;          // m: tokens per expert the program is built for (dispatch capacity)
    bool weights_bf8 = false;         // weight tiles bfp8 (else bfp4)
    uint32_t activation = 0;          // se3_compute.cpp SE_ACT
    uint32_t pin = 1;                 // SE_PIN_MIN (0: no pinning)
    bool y_row_major = false;         // y as row-major bf16 [rows, H] (else bfp8 tiles); no effect on the plan

    static constexpr auto attribute_names = std::forward_as_tuple(
        "hidden",
        "intermediate",
        "experts_per_chip",
        "num_global_experts",
        "max_tokens",
        "weights_bf8",
        "activation",
        "pin",
        "y_row_major");
    auto attribute_values() const {
        return std::forward_as_tuple(
            hidden,
            intermediate,
            experts_per_chip,
            num_global_experts,
            max_tokens,
            weights_bf8,
            activation,
            pin,
            y_row_major);
    }
};

// The spatial pipeline's layout: role cores, the work split and the per-core L1 arena. A pure function of the
// device grid and the config (the program factory, the host wrapper's scratch tensors and the weight layout all use
// the same plan). Mirrors the model path (dynamic counts, row-major x, helper relays) of the research builder
// models/demos/mimo_v2_d_p/tt/flat_expert.py; see tests/perf/FLAT_EXPERT_WORKLOG.md there.
struct FlatRoutedExpertPlan {
    using Core = tt::tt_metal::CoreCoord;
    // shapes
    uint32_t H = 0, I = 0, E = 0, NG = 0, m = 0, Ht = 0, It = 0;
    uint32_t w_tile = 0;  // weight tile bytes
    uint32_t banks = 0;
    // subgrids, gate/up split, sub-blocks
    uint32_t nsg = 1, n_rd_sg = 16;
    uint32_t np = 1, g = 1, mt = 1, mtg = 1, x_slots = 24, dst_tiles = 8;
    bool gu_rp = false, gu_fp32 = true, gu_l1acc = false;
    uint32_t s = 1, v = 1, nk_gu = 1, slot = 1, rg = 1, r_ = 1, ring_g = 1;
    // relays
    uint32_t nh = 1, land_slots = 3, vstride = 2, sbt = 32, seg = 2048, nsb = 1;
    bool group_rect = false;
    // down
    bool rdown = false;
    uint32_t d_ch = 7, n_rdn = 0, pcd_r = 0, rem_cols = 0, kd_r = 0, nblk_r = 0, slot_dr = 0, ring_dr = 0;
    uint32_t out_tiles_r = 0, h_tiles = 0, hbuf = 3, nd_sg = 1;
    float dring = 2.0f;
    std::vector<uint32_t> pcds, col0s;
    // arena (per core, 2 KB aligned offsets)
    uint32_t x_off = 0, p_off = 0, rd_off = 0, h_off = 0, o_off = 0, sb_off = 0, land_off = 0, arena_tiles = 0;
    uint32_t rd_slots = 2;
    // weight regions (bytes per core region)
    uint32_t region_bytes = 0, wd_region = 0, wr_region = 0;
    // role cores (logical)
    std::vector<Core> readers, gu, gu_idle, relays, down, coords;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> rects;  // x0, x1, y0, y1
    std::vector<std::pair<uint32_t, uint32_t>> rdn;                         // (reader, chain tail down core)
    std::map<uint32_t, uint32_t> d_pred, d_succ;
    std::vector<uint32_t> d_heads;

    std::vector<Core> arena_cores() const;  // gu + down + relays + readers + gu_idle
    uint32_t sg_rd(uint32_t r) const { return r / n_rd_sg; }
    uint32_t sg_dn(uint32_t d) const { return d / nd_sg; }
    uint32_t dl(uint32_t d) const { return d % nd_sg; }
    uint32_t rect_of(const Core& c) const;
    static uint32_t kd_of(uint32_t pcd, uint32_t it);
};

FlatRoutedExpertPlan make_flat_routed_expert_plan(tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& cfg);

// Greedy maximal rectangles over a core list (one range per core makes every dispatch write a unicast).
tt::tt_metal::CoreRangeSet rect_ranges(const std::vector<tt::tt_metal::CoreCoord>& cores);

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
