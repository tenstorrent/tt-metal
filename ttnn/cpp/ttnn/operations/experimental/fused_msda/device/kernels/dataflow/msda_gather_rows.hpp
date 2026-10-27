// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-major gather of one sampling point's four bilinear corners, for a range of
// query rows. Shared by the reader and, when the gather is split, by the writer:
// the two data-movement RISCs each take a disjoint range of rows of the same
// staging block, so they share the issue cost while every row is still produced
// by exactly one RISC.
//
// Row r of the block is [NW | NE | SW | SE], D bf16 each (see
// fused_msda_reader_common.hpp). Only in-bounds corners of rows < v_rows are
// read. Compute zeroes the weight of every out-of-bounds corner, so on
// Blackhole a skipped slot keeps whatever the CB held: zero from startup or an
// earlier value stick, which the zero weight cancels even when it is inf or
// NaN, because Blackhole's FPU multiplies both by zero to zero. Elsewhere that
// has not been measured, so the gathers also zero the out-of-bounds slots
// (ZERO_SKIPPED_SLOTS). Rows >= v_rows are never read or written: their weights
// can be live (tiled geometry), but the writer emits only rows < v_rows.
// The caller barriers its own NoC reads; nothing here waits.

#pragma once

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_corner_word.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"

namespace fused_msda_gather {

#ifdef ARCH_BLACKHOLE
constexpr bool ZERO_SKIPPED_SLOTS = false;
#else
constexpr bool ZERO_SKIPPED_SLOTS = true;
#endif

// Everything one point's gather needs, as the reader hands it to the writer
// through the mailbox. Plain uint32 words so the mailbox is a flat L1 copy.
struct PointArgs {
    // Row r's corner word at corner_l1 + tile_col16_offset_fp32(r): a corner
    // tile, or a tiled block's shifted by j columns.
    uint32_t corner_l1;
    uint32_t block_l1;  // the input-CB block being staged
    uint32_t v_rows;
    uint32_t start_index;  // level_start of this point's level
    uint32_t width;
    uint32_t batch;
    uint32_t head;
};
constexpr uint32_t POINT_ARGS_WORDS = sizeof(PointArgs) / sizeof(uint32_t);

// Split gather: the reader takes rows [0, SPLIT_ROW), the writer the rest. A
// tail block with v_rows <= SPLIT_ROW leaves the writer nothing to read.
//
// Empirical: op device time on Blackhole, tiled geometry, D = 32 (SCA in
// test_encoder_layer_perf.py bev_100x100 and nuscenes_base in
// test_fused_msda_perf.py); a row either way is 1-10% slower. The time is not
// linear in the split, so re-measure rather than derive it. Both RISCs take
// the row from the same accessor type, so they agree on which rows they own.
// test_fused_msda_v1_masks_out_of_bounds_corners puts v_rows on the split
// through its own copy of these values; change both together.
constexpr uint32_t SPLIT_ROW_L1 = 16;
constexpr uint32_t SPLIT_ROW_DRAM = 15;
static_assert(SPLIT_ROW_L1 > 0 && SPLIT_ROW_L1 < fused_msda_tile_layout::TILE_MAX_ROWS);
static_assert(SPLIT_ROW_DRAM > 0 && SPLIT_ROW_DRAM < fused_msda_tile_layout::TILE_MAX_ROWS);
template <typename ValueAccessor>
constexpr uint32_t SPLIT_ROW = ValueAccessor::is_dram ? SPLIT_ROW_DRAM : SPLIT_ROW_L1;

// Split-gather handshake. Per point, in the same (tile, level, point) order on
// both RISCs:
//   reader: post_point(pt); ready = ++seq; push the next point's geometry
//           (per-point hand-off only); gather rows [0, SPLIT_ROW); barrier;
//           wait done == seq; pop the corner tile (tiled geometry: after the
//           block's last point); push the block.
//   writer: wait ready == ++seq; fetch_point; gather rows [SPLIT_ROW, 32);
//           barrier; done = seq.
// Both sides must post or consume exactly NUM_LEVELS * NUM_POINTS points per
// output tile, every tile: a skipped post hangs the other side. The mailbox is
// rewritten only after done == seq, so one page is enough. These are counters
// rather than a CB because the writer fills part of a block the reader
// reserved, and the corner tile must stay at the reader's front until the
// writer has read it.
//
// Ordering: volatile accesses keep the mailbox stores ahead of ready. The L1
// cache note in internal/tt-1xx/cache.h covers visibility (the waiter
// invalidates on every poll).

inline void post_point(uint32_t mailbox_l1, const PointArgs& a) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&a);
    CoreLocalMem<volatile uint32_t> dst(mailbox_l1);
    for (uint32_t i = 0; i < POINT_ARGS_WORDS; ++i) {
        dst[i] = src[i];
    }
}

inline PointArgs fetch_point(uint32_t mailbox_l1) {
    PointArgs a;
    uint32_t* dst = reinterpret_cast<uint32_t*>(&a);
    CoreLocalMem<volatile uint32_t> src(mailbox_l1);
    for (uint32_t i = 0; i < POINT_ARGS_WORDS; ++i) {
        dst[i] = src[i];
    }
    return a;
}

inline void zero_words(uint32_t l1_addr, uint32_t words) {
    uint32_t* p = &CoreLocalMem<uint32_t>(l1_addr)[0];
#pragma GCC unroll 8
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0;
    }
}

// Cfg carries the value layout as static constexpr members: D, NUM_KEYS,
// NUM_HEADS, VALUE_PACKED (packed (B, S, H*D) vs canonical (B, S, H, D)) and
// STICK_NBYTES. The stick must be exactly D*2 bytes (row-major staging packs
// the four slots back to back).
//
// Kept out of line: the reader's main loop keeps far more values live than the
// writer's, and inlined there the corner loop spills its NoC register pointers
// and level fields to the stack. Everything the loop reads is copied into
// locals: the NoC command-register stores are volatile, so a field read through
// a reference would be reloaded on every read.
template <typename Cfg, typename ValueAccessor>
__attribute__((noinline)) void gather_rows(
    Noc& caller_noc, const ValueAccessor& value_acc_ref, const PointArgs& a, uint32_t r_begin, uint32_t r_end) {
    Noc noc(caller_noc.get_noc_id());
    const ValueAccessor value_acc = value_acc_ref;
    const uint32_t block_l1 = a.block_l1;
    const uint32_t corner_l1 = a.corner_l1;
    const uint32_t v_rows = a.v_rows;
    const uint32_t width = a.width;
    const uint32_t head = a.head;
    constexpr uint32_t STICK = Cfg::STICK_NBYTES;
    constexpr uint32_t ROW_NBYTES = 4 * STICK;
    constexpr bool PACKED = Cfg::VALUE_PACKED;

    namespace cw = fused_msda_corner_word;
    // (b, s) row of the NW corner is nw_base + index. It wraps below zero only
    // for an out-of-bounds NW, which is never read.
    const uint32_t nw_base = a.batch * Cfg::NUM_KEYS + a.start_index - (width + 1);

    // Rows >= v_rows never reach the output.
    const uint32_t r_last = r_end < v_rows ? r_end : v_rows;
    for (uint32_t r = r_begin; r < r_last; ++r) {
        const uint32_t row_l1 = block_l1 + r * ROW_NBYTES;
        const auto c = cw::decode(
            CoreLocalMem<volatile uint32_t>(corner_l1 + fused_msda_tile_layout::tile_col16_offset_fp32(r))[0]);
        const uint32_t nw = nw_base + c.index;

        // NW, NE, SW, SE, in lockstep with the four scalar packs in
        // msda_geometry.hpp::point.
        for (uint32_t dy = 0; dy < 2; ++dy) {
            if (!(c.flags & (cw::FLAG_Y0 << dy))) {
                if constexpr (ZERO_SKIPPED_SLOTS) {
                    zero_words(row_l1 + 2u * dy * STICK, 2 * STICK / sizeof(uint32_t));
                }
                continue;
            }
            const uint32_t pair_l1 = row_l1 + 2u * dy * STICK;
            for (uint32_t dx = 0; dx < 2; ++dx) {
                if (!(c.flags & (cw::FLAG_X0 << dx))) {
                    if constexpr (ZERO_SKIPPED_SLOTS) {
                        zero_words(pair_l1 + dx * STICK, STICK / sizeof(uint32_t));
                    }
                    continue;
                }
                const uint32_t slot_l1 = pair_l1 + dx * STICK;
                const uint32_t bs = nw + dy * width + dx;
                uint32_t page;
                uint32_t offset_bytes;
                if constexpr (PACKED) {
                    page = bs;
                    offset_bytes = head * (Cfg::D * 2u);
                } else {
                    page = bs * Cfg::NUM_HEADS + head;
                    offset_bytes = 0;
                }
                noc.async_read(
                    value_acc,
                    CoreLocalMem<uint32_t>(slot_l1),
                    STICK,
                    {.page_id = page, .offset_bytes = offset_bytes},
                    {.offset_bytes = 0});
            }
        }
    }
}

}  // namespace fused_msda_gather
