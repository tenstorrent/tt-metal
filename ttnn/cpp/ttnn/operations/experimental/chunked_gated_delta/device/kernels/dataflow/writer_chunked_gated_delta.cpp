// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per (head, seq_idx) the writer:
//   1) Async-writes each tile of output_cb to its DRAM page in the global output.
//   2) For all seq_idx except the last in the head, also issues a local L1-to-L1
//      NOC copy of the same output_cb tiles into current_state_cb. This feeds the
//      compute kernel's next-iteration state without making compute the producer
//      of state_cb (which would create a two-writer hazard against the reader).
//   3) Barriers on the writes, then push current_state_cb (when filled) and pop
//      output_cb so compute can continue.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/device_print.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    // Per-core slice of the global head dimension. ``num_heads`` is how many heads
    // this core is responsible for; ``head_offset`` is the global index of the
    // first head, used to compute absolute DRAM page IDs.
    const uint32_t num_heads = get_arg_val<uint32_t>(1);
    const uint32_t head_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t current_state_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t seq_len = get_compile_time_arg_val(2);
    constexpr uint32_t dim_k = get_compile_time_arg_val(3);
    constexpr uint32_t dim_v = get_compile_time_arg_val(4);

    constexpr auto output_args = TensorAccessorArgs<5>();

    constexpr uint32_t tile_hw = 32;
    constexpr uint32_t state_ht = dim_k / tile_hw;
    constexpr uint32_t state_wt = dim_v / tile_hw;
    constexpr uint32_t state_tiles_per_step = state_ht * state_wt;

    DEVICE_PRINT("num_heads: {}, head_offset: {}\n", num_heads, head_offset);
    DEVICE_PRINT("seq_len: {}\n", seq_len);
    DEVICE_PRINT("dim_k: {}\n", dim_k);
    DEVICE_PRINT("dim_v: {}\n", dim_v);
    DEVICE_PRINT("state_tiles_per_step: {}\n", state_tiles_per_step);
    DEVICE_PRINT("state_ht: {}\n", state_ht);
    DEVICE_PRINT("state_wt: {}\n", state_wt);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer output_cb(output_cb_index);
    CircularBuffer current_state_cb(current_state_cb_index);

    const uint32_t tile_bytes = get_tile_size(output_cb_index);
    const uint32_t state_bytes = state_tiles_per_step * tile_bytes;
    const uint8_t noc_id = noc.get_noc_id();

    for (uint32_t head_local = 0; head_local < num_heads; ++head_local) {
        const uint32_t head = head_local + head_offset;
        for (uint32_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            const uint32_t output_page_base = head * seq_len * state_tiles_per_step + seq_idx * state_tiles_per_step;
            const bool feed_state = (seq_idx + 1) < seq_len;

            // Wait for the matmul output produced by compute for this (head, seq_idx).
            output_cb.wait_front(state_tiles_per_step);

            // Reserve the recurrent-state slot up front so the local L1-to-L1 NOC copy
            // below has a guaranteed destination range. Skipped on the final seq_idx
            // of the head since compute does not need a follow-on state.
            if (feed_state) {
                current_state_cb.reserve_back(state_tiles_per_step);
            }

            // 1) Stream tiles to DRAM.
            for (uint32_t tile = 0; tile < state_tiles_per_step; ++tile) {
                noc.async_write(
                    output_cb,
                    output_accessor,
                    tile_bytes,
                    {.offset_bytes = tile * tile_bytes},
                    {.page_id = output_page_base + tile});
            }

            // 2) Local L1-to-L1 NOC copy: output_cb -> current_state_cb on this core.
            //    A single noc_async_write moves the entire state_tiles_per_step tile
            //    block (the underlying API will split into NOC packets as needed).
            if (feed_state) {
                const uint32_t src_l1_addr = output_cb.get_read_ptr();
                const uint32_t dst_l1_addr = current_state_cb.get_write_ptr();
                const uint64_t dst_noc_addr = get_noc_addr(my_x[noc_id], my_y[noc_id], dst_l1_addr, noc_id);
                noc_async_write(src_l1_addr, dst_noc_addr, state_bytes, noc_id);
            }

            // Single barrier covers both the DRAM stream-out and the local self-copy.
            noc.async_write_barrier();

            if (feed_state) {
                current_state_cb.push_back(state_tiles_per_step);
            }
            output_cb.pop_front(state_tiles_per_step);
        }
    }
}
