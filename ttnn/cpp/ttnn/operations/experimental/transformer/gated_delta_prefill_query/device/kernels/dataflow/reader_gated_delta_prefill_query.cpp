// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the gated-delta prefill-then-query op.
//
// Each core owns one section of one V-head's K/V prefill sequence. It reads K first, then V:
//   * K is replicated across a GVA group, so this core reads K-head `k_head_id` (= v_head /
//     gva_ratio) into cb_k.
//   * V is per V-head, so this core reads its own `v_head_id` into cb_v, AFTER all of K is in.
// Both are streamed one tile at a time in hidden-major order over the core's seq-tile range
// [seq_tile_start, seq_tile_start + seq_tile_count): for each hidden tile, the section's seq-tiles
// are pushed contiguously (cb[kd * seq_tile_count + s]). Pushing per tile lets compute start as
// each hidden tile lands; reads are strided in DRAM — fine here.
//
// K/V are tiled [1, N, S, d]: within a head the [S, d] matrix is seq-tile-major, hidden-tile-minor,
// so the tile page for (head, seq_tile i, hidden_tile j) is
//   page = head * (seq_tiles * d_tiles) + i * d_tiles + j.

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
    const uint32_t v_head_id = get_arg_val<uint32_t>(4);
    // Group metadata (args 5,6): section_id, num_sections — reserved for the future per-V-head
    // tree reduction; unused here.
    [[maybe_unused]] const uint32_t section_id = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t num_sections = get_arg_val<uint32_t>(6);
    const uint32_t v_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t d_tiles = get_compile_time_arg_val(0);    // hidden-dim width, in tiles
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(1);  // full K/V seq length, in tiles

    constexpr uint32_t cb_k = tt::CBIndex::c_0;
    constexpr uint32_t cb_v = tt::CBIndex::c_6;
    const uint32_t tile_bytes = get_tile_size(cb_k);

    constexpr auto k_args = TensorAccessorArgs<2>();
    const auto k_gen = TensorAccessor(k_args, k_addr, tile_bytes);
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto v_gen = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_k_o(cb_k);
    CircularBuffer cb_v_o(cb_v);

    // ---- Pass 1: K for k_head_id (replicated across the GVA group), hidden-major. ----
    const uint32_t k_head_page_base = k_head_id * seq_tiles * d_tiles;
    for (uint32_t kd = 0; kd < d_tiles; ++kd) {
        for (uint32_t s = 0; s < seq_tile_count; ++s) {
            const uint32_t page = k_head_page_base + (seq_tile_start + s) * d_tiles + kd;
            cb_k_o.reserve_back(1);
            noc.async_read(k_gen, cb_k_o, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_k_o.push_back(1);
        }
    }

    // ---- Pass 2: V for v_head_id (this core's own V-head), hidden-major, after all of K. ----
    const uint32_t v_head_page_base = v_head_id * seq_tiles * d_tiles;
    for (uint32_t kd = 0; kd < d_tiles; ++kd) {
        for (uint32_t s = 0; s < seq_tile_count; ++s) {
            const uint32_t page = v_head_page_base + (seq_tile_start + s) * d_tiles + kd;
            cb_v_o.reserve_back(1);
            noc.async_read(v_gen, cb_v_o, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_v_o.push_back(1);
        }
    }
}
