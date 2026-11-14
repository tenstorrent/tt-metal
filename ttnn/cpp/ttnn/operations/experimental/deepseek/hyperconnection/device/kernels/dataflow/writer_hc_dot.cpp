// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

// A fn dot core below the root sends its reduced partial up one level of the tree: straight into its slot in the
// parent's cb_recv, then one credit on the parent's semaphore for that level.
//
// The root pushes the finished [1, N] fused_w row into cb_fused_w on every core that reads it, then raises
// fused_w_sem there. cb_fused_w spans the root and those cores, so it sits at the same L1 address on all of them.
void kernel_main() {
    constexpr uint32_t send_cb_index = get_named_compile_time_arg_val("cb_send");
    constexpr uint32_t recv_cb_index = get_named_compile_time_arg_val("cb_recv");
    constexpr uint32_t out_cb_index = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t fused_w_cb_index = get_named_compile_time_arg_val("cb_fused_w");

    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t fused_w_sem = get_compile_time_arg_val(4);

    const uint32_t has_parent = get_arg_val<uint32_t>(0);
    Noc noc;

    if (has_parent) {
        const uint32_t parent_noc_x = get_arg_val<uint32_t>(1);
        const uint32_t parent_noc_y = get_arg_val<uint32_t>(2);
        const uint32_t parent_level = get_arg_val<uint32_t>(3);
        const uint32_t slot_offset_bytes = get_arg_val<uint32_t>(4);

        CircularBuffer send_cb(send_cb_index);
        CircularBuffer recv_cb(recv_cb_index);
        Semaphore<> parent_arrived(parent_level);
        UnicastEndpoint parent;

        // This RISC never advances cb_recv, so its write pointer is the CB's base -- the same L1
        // address as on the parent.
        const uint32_t dst_addr = recv_cb.get_write_ptr() + slot_offset_bytes;

        send_cb.wait_front(block_num_tiles);
        noc.async_write(
            send_cb,
            parent,
            block_bytes,
            {.offset_bytes = 0},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = dst_addr});
        // The partial must have landed before the parent can see the credit.
        noc.async_write_barrier();
        parent_arrived.up(noc, parent_noc_x, parent_noc_y, 1);
        noc.async_atomic_barrier();
        send_cb.pop_front(block_num_tiles);
        return;
    }

    const uint32_t num_dests = get_arg_val<uint32_t>(5);
    constexpr uint32_t dests_arg = 6;
    CircularBuffer out_cb(out_cb_index);
    // Never advanced here, so this is the CB's base on every core it spans.
    const uint32_t dst_addr = CircularBuffer(fused_w_cb_index).get_write_ptr();

    out_cb.wait_front(out_num_tiles);
    const uint32_t src_addr = out_cb.get_read_ptr();
    for (uint32_t i = 0; i < num_dests; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(dests_arg + 2 * i);
        const uint32_t y = get_arg_val<uint32_t>(dests_arg + 2 * i + 1);
        noc_async_write(src_addr, get_noc_addr(x, y, dst_addr), out_bytes);
    }
    // The row must have landed before a reader can see the credit.
    noc.async_write_barrier();
    Semaphore<> landed(fused_w_sem);
    for (uint32_t i = 0; i < num_dests; ++i) {
        landed.up(noc, get_arg_val<uint32_t>(dests_arg + 2 * i), get_arg_val<uint32_t>(dests_arg + 2 * i + 1), 1);
    }
    noc.async_atomic_barrier();
    out_cb.pop_front(out_num_tiles);
}
