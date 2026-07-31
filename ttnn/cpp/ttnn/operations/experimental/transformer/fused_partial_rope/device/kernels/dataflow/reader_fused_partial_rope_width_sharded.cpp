// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Reader for the width-sharded fused partial-RoPE path. X and the output are resident L1 shards;
// cos / sin / trans_mat are DRAM-interleaved. This core owns every row but only a column slice of
// the head dim, of which the trailing `rope_local` tiles fall in the rope region (starting at
// column tile `rope_col_start` of the cos/sin tables). So it streams, for each of its Ht row-tiles,
// that row's `rope_local` cos + sin tiles, plus the single (replicated) trans_mat tile.
//
// A core whose columns are entirely in the "nope" region has no rope work at all: it reads nothing.
void kernel_main() {
    uint32_t argrt = 0;
    const uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t rope_local = get_arg_val<uint32_t>(argrt++);
    const uint32_t rope_col_start = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    // Width (in tiles) of the cos/sin tables, i.e. the page stride between row-tiles.
    constexpr uint32_t rope_Wt_total = get_compile_time_arg_val(4);
    // When set, cos/sin hold a single tile-row that the compute kernel broadcasts across all rows.
    constexpr bool cos_bcast = get_compile_time_arg_val(5) != 0;
    constexpr auto cos_args = TensorAccessorArgs<6>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cos_rows_t = cos_bcast ? 1 : Ht;

    if (rope_local == 0) {
        return;
    }

    Noc noc;
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer trans_mat_cb(trans_mat_cb_id);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
    const auto s_cos = TensorAccessor(cos_args, cos_addr);
    const auto s_sin = TensorAccessor(sin_args, sin_addr);
    const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr);

    // trans_mat: single replicated tile (page 0), reused for the whole compute.
    trans_mat_cb.reserve_back(onetile);
    noc.async_read(
        s_trans_mat, CoreLocalMem<uint32_t>(trans_mat_cb.get_write_ptr()), trans_mat_tile_bytes, {.page_id = 0}, {});

    // cos / sin: this core's rope columns for every row-tile it owns, laid out row-major so the
    // compute kernel indexes tile (rt, j) at rt * rope_local + j.
    const uint32_t cos_sin_tiles = rope_local * cos_rows_t;
    cos_cb.reserve_back(cos_sin_tiles);
    sin_cb.reserve_back(cos_sin_tiles);
    uint32_t cos_l1 = cos_cb.get_write_ptr();
    uint32_t sin_l1 = sin_cb.get_write_ptr();
    for (uint32_t rt = 0; rt < cos_rows_t; ++rt) {
        const uint32_t row_page = rt * rope_Wt_total + rope_col_start;
        for (uint32_t j = 0; j < rope_local; ++j) {
            noc.async_read(s_cos, CoreLocalMem<uint32_t>(cos_l1), cos_tile_bytes, {.page_id = row_page + j}, {});
            noc.async_read(s_sin, CoreLocalMem<uint32_t>(sin_l1), sin_tile_bytes, {.page_id = row_page + j}, {});
            cos_l1 += cos_tile_bytes;
            sin_l1 += sin_tile_bytes;
        }
    }

    noc.async_read_barrier();
    trans_mat_cb.push_back(onetile);
    cos_cb.push_back(cos_sin_tiles);
    sin_cb.push_back(cos_sin_tiles);
}
