// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the gated-delta prefill-then-query op.
//
// Each core owns one section of one V-head's K/V prefill sequence. K is replicated across a
// GVA group, so this core reads K-head `k_head_id` (= v_head / gva_ratio). It streams its
// seq-tile range [seq_tile_start, seq_tile_start + seq_tile_count) into cb_k one seq-row at a
// time: each push is `d_tiles` tiles — a single seq-tile spanning the FULL hidden dim — so the
// compute consumer can start working on a row as soon as it lands. The cb_k capacity
// (out_block_size = block_height * d_tiles tiles) sets how many rows can be in flight before
// the reader blocks on the consumer.
//
// K is tiled [1, Nk, S, d]: within a head the [S, d] matrix is laid out seq-tile-major,
// hidden-tile-minor, so the tile page for (k_head, seq_tile i, hidden_tile j) is
//   page = k_head * (seq_tiles * d_tiles) + i * d_tiles + j.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_head_id = get_arg_val<uint32_t>(1);
    const uint32_t seq_tile_start = get_arg_val<uint32_t>(2);
    const uint32_t seq_tile_count = get_arg_val<uint32_t>(3);
    // Group metadata (args 4,5,6): v_head_id, section_id, num_sections — reserved for the
    // future per-V-head tree reduction; unused by this K read.
    [[maybe_unused]] const uint32_t v_head_id = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t section_id = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t num_sections = get_arg_val<uint32_t>(6);

    constexpr uint32_t d_tiles = get_compile_time_arg_val(0);    // hidden-dim width, in tiles
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(1);  // full K/V seq length, in tiles

    constexpr uint32_t cb_k = tt::CBIndex::c_0;
    const uint32_t tile_bytes = get_tile_size(cb_k);

    constexpr auto k_args = TensorAccessorArgs<2>();
    const auto k_gen = TensorAccessor(k_args, k_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_k_o(cb_k);

    const uint32_t head_page_base = k_head_id * seq_tiles * d_tiles;

    // One push per seq-row: d_tiles tiles (a full-hidden-dim seq-tile). Capacity gates depth.
    for (uint32_t row = 0; row < seq_tile_count; ++row) {
        const uint32_t page_row_base = head_page_base + (seq_tile_start + row) * d_tiles;
        cb_k_o.reserve_back(d_tiles);
        for (uint32_t j = 0; j < d_tiles; ++j) {
            noc.async_read(k_gen, cb_k_o, tile_bytes, {.page_id = page_row_base + j}, {.offset_bytes = j * tile_bytes});
        }
        noc.async_read_barrier();
        cb_k_o.push_back(d_tiles);
    }
}
