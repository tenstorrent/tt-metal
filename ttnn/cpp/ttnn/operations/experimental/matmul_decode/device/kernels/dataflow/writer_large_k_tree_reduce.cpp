// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Sends this core's reduced partial up one level of the tree: straight into its slot in the
// parent's cb_recv, then one credit on the parent's semaphore for that level. The root has no
// parent and does nothing.
void kernel_main() {
    constexpr uint32_t send_cb_index = get_named_compile_time_arg_val("cb_send");
    constexpr uint32_t recv_cb_index = get_named_compile_time_arg_val("cb_recv");

    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(1);

    const uint32_t has_parent = get_arg_val<uint32_t>(0);
    if (!has_parent) {
        return;
    }
    const uint32_t parent_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t parent_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t parent_level = get_arg_val<uint32_t>(3);
    const uint32_t slot_offset_bytes = get_arg_val<uint32_t>(4);

    Noc noc;
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
}
