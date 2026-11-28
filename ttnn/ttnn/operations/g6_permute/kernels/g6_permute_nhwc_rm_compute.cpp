// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute compute (nhwc_rm_transpose). A unit is kwu w-tiles of cb_input_slots; w-tile wt is a
// 32-row row-major block [w][c] of ct tiles. Per unit:
//   tilize_unit:   the kwu blocks -> kwu * ct tiles in cb_tiles (tile (wt, ct) at wt * ct + ct).
//   transpose_unit: per c-tile, tiles (0..kwu-1, ct) transposed into DEST slots 0..kwu-1, then
//                   pack_untilize_dest writes them as 32 rows c of kwu * 32 values w: one cb_output_rows
//                   page per c-tile.
// Every CB is UInt16, so unpack and pack move the 16 bits opaquely: no value is converted.
//
// Helper bypass: tilize_block / transpose_tile / pack_untilize_dest, not kernel_lib. The tilize helper
// waits for and pops its input per tile-row block, but a unit's w-tiles share one CB page when the
// shared DRAM reader pushes one page per unit, so the tilize must address w-tile wt at an offset from
// the read pointer. kernel_lib has no transpose helper, and its untilize helper has no transposing
// unpack.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
constexpr uint32_t cb_tiles = get_compile_time_arg_val(2);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(3);
constexpr uint32_t kw = get_compile_time_arg_val(4);
constexpr uint32_t kw_tail = get_compile_time_arg_val(5);
constexpr uint32_t units_per_row = get_compile_time_arg_val(6);
// w-tiles per cb_input_slots page: kw with the shared DRAM reader (one page per unit), 1 with the L1
// reader. A unit is always unit_pages pages.
constexpr uint32_t in_page_tiles = get_compile_time_arg_val(7);
constexpr uint32_t unit_pages = kw / in_page_tiles;
static_assert(kw % in_page_tiles == 0);
static_assert(kw >= 1 && kw <= 8 && kw_tail >= 1 && kw_tail <= kw, "a unit's w-tiles fit one half of 16-bit DEST");

template <uint32_t kwu>
ALWI void tilize_unit() {
    tilize_init(cb_input_slots, tensor_ct, cb_tiles);
    cb_reserve_back(cb_tiles, kwu * tensor_ct);
    for (uint32_t wt = 0; wt < kwu; ++wt) {
        // Cumulative wait: each block starts as soon as its own w-tile has landed.
        cb_wait_front(cb_input_slots, wt / in_page_tiles + 1);
        tilize_block(cb_input_slots, tensor_ct, cb_tiles, wt * tensor_ct, wt * tensor_ct);
    }
    cb_push_back(cb_tiles, kwu * tensor_ct);
    // A tail unit's pad pages are popped with it.
    cb_wait_front(cb_input_slots, unit_pages);
    cb_pop_front(cb_input_slots, unit_pages);
    tilize_uninit(cb_input_slots, cb_tiles);
}

template <uint32_t kwu>
ALWI void transpose_unit() {
    transpose_init(cb_tiles);
    pack_untilize_dest_init<kwu, kwu>(cb_output_rows);
    cb_wait_front(cb_tiles, kwu * tensor_ct);
    for (uint32_t ct = 0; ct < tensor_ct; ++ct) {
        tile_regs_acquire();
        for (uint32_t wt = 0; wt < kwu; ++wt) {
            transpose_tile(cb_tiles, wt * tensor_ct + ct, wt);
        }
        tile_regs_commit();
        cb_reserve_back(cb_output_rows, 1);
        tile_regs_wait();
        pack_untilize_dest<kwu, kwu>(cb_output_rows);
        tile_regs_release();
        cb_push_back(cb_output_rows, 1);
    }
    cb_pop_front(cb_tiles, kwu * tensor_ct);
    pack_untilize_uninit(cb_output_rows);
}

// Sum zones (~5 cycles a step, no marker cap) cover every unit; each includes its CB waits.
template <uint32_t kwu>
ALWI void process_unit() {
    {
        DeviceZoneScopedSumN1("PERM-CP-TILIZE");
        tilize_unit<kwu>();
    }
    {
        DeviceZoneScopedSumN2("PERM-CP-TRANSPOSE");
        transpose_unit<kwu>();
    }
}

ALWI void process(uint32_t row_pos) {
    if (kw_tail != kw && row_pos == units_per_row - 1) {
        process_unit<kw_tail>();
    } else {
        process_unit<kw>();
    }
}

void kernel_main() {
    uint32_t remaining = get_arg_val<uint32_t>(0);
    uint32_t row_pos = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_input_slots, cb_output_rows);
    // The first unit alone is a scoped zone: it shows the pipeline fill.
    for (bool first = true; remaining > 0; --remaining, first = false) {
        if (first) {
            DeviceZoneScopedN("PERM-CP-FIRST-UNIT");
            process(row_pos);
        } else {
            process(row_pos);
        }
        if (++row_pos == units_per_row) {
            row_pos = 0;
        }
    }
}
