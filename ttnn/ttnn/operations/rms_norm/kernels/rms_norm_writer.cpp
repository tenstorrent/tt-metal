// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1) — op_design.md §4.1.
//
// Mirrors the reader's row-block/chunk nest exactly: per core, per row-block of
// HT_BLOCK tile-rows, drain NW chunks of WT_CHUNK W-tiles. The TILE path writes
// whole tile pages with one barrier per chunk (the batched twin of the reader's
// batched chunk read); the ROW_MAJOR path goes through
// dataflow_kernel_lib::write_sticks_after_untilize, which skips the L1 W-padding
// and writes only the valid sticks of a short last tile-row — that is what makes
// non-aligned H/W native with no host-side slice.
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_write_page directly because write_sticks_after_untilize writes
// *sticks* produced by the untilize helper, and the TILE path never untilizes —
// its input contract cannot be met. op_design.md §7 mandates exactly this.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    // args 1..5 are the reader/compute regime flags; kept in one shared layout.
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // ---- geometry ----
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(12);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(16);

    constexpr auto out_args = TensorAccessorArgs<17>();

    static_assert(WT_LAST == WT_CHUNK, "writer assumes uniform chunk widths");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        const uint32_t row0 = start_tile_row + hb * HT_BLOCK;
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }

        uint32_t valid_rows = ht * 32u;
        if constexpr (IS_RM) {
            const uint32_t remaining = TOTAL_STICKS - row0 * 32u;
            if (remaining < valid_rows) {
                valid_rows = remaining;
            }
        }

        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM) {
                const uint32_t rb = (wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
                dkl::write_sticks_after_untilize<cb_output_rm>(
                    out_acc, valid_rows, rb, row0 * 32u, wc * CHUNK_ROW_BYTES);
            } else {
                const uint32_t n = ht * WT_CHUNK;
                cb_wait_front(cb_output_tiles, n);
                uint32_t addr = get_read_ptr(cb_output_tiles);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t base_tile = (row0 + h) * WT + wc * WT_CHUNK;
                    for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                        noc_async_write_page(base_tile + t, out_acc, addr);
                        addr += out_tile_bytes;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, n);
            }
        }
    }
}
