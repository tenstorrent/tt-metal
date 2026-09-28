// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter_average — relay writer (BRISC), cores (0,0) [forward] and (0,1)
// [backward], direction selected by CT arg.
//
// Fabric egress half of the line store-and-forward gather, through the
// safety-by-construction CCL helper (FabricStreamSender -> FabricStream ->
// UnicastWriteChannel / AtomicIncChannel). For chip i, direction d (0 = forward ->
// i+1, 1 = backward -> i-1), the writer sends num_sends blocks one hop:
//   * k == 0: seed block i (this device's own shard).
//   * k >= 1: relay block (i -/+ k) mod N — the block the relay reader just read
//     back out of the local gather_buffer.
// Every page lands DIRECTLY in the downstream device's gather_buffer at the block's
// canonical page range c*P + p (uniform mesh buffer address, local accessor routed
// one hop). After each block's last page the writer issues TWO counting atomic-incs
// on the SAME connection (T4): sem_dir at the receiving RELAY core (store-and-forward
// wait) and sem_dir at the receiving REDUCE core (the overlap trigger — the reduce
// pass for this block starts while the next block is still in flight, T7). Fabric
// delivery is in-order per connection, so sem >= k implies blocks 1..k landed (R8).
//
// An idle direction (num_sends == 0: line ends) opens no connection and reads NO
// runtime args (its rt-arg list is empty). noc_async_writes_flushed() before every
// cb_pop_front — the fabric write sources the CB slot (R7).
//
// CT args: [cb_relay_pages, direction, my_chip_id, ring_size, num_sends,
//           l1_alignment] + gather TensorAccessorArgs
// RT args (num_sends > 0 only): [fabric conn block (FIRST — cursor from 0)]
//           [gather_buffer_addr, pages_per_shard, page_size, num_hops, sem_addr,
//            relay_noc_x, relay_noc_y, reduce_noc_x, reduce_noc_y]

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t direction = get_compile_time_arg_val(1);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_sends = get_compile_time_arg_val(4);
    constexpr uint32_t alignment = get_compile_time_arg_val(5);
    constexpr auto gather_buffer_args = TensorAccessorArgs<6>();

    // Idle direction: no fabric egress, no connection, no rt args.
    if constexpr (num_sends > 0) {
        // Fabric connection block FIRST (build_ccl_fabric_rt_args layout); its leading
        // has_forward flag also encodes the send direction, which we peek.
        size_t conn_arg_idx = 0;
        const bool dst_is_forward = get_arg_val<uint32_t>(0) == 1;
        FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

        // Op args resume at the cursor the sender advanced past the fabric block.
        uint32_t ai = static_cast<uint32_t>(conn_arg_idx);
        const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
        const uint32_t page_size = get_arg_val<uint32_t>(ai++);
        const uint32_t num_hops = get_arg_val<uint32_t>(ai++);
        const uint32_t sem_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t relay_noc_x = get_arg_val<uint32_t>(ai++);
        const uint32_t relay_noc_y = get_arg_val<uint32_t>(ai++);
        const uint32_t reduce_noc_x = get_arg_val<uint32_t>(ai++);
        const uint32_t reduce_noc_y = get_arg_val<uint32_t>(ai++);

        const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
        const uint32_t P = pages_per_shard;

        auto stream = sender.open(unicast_route(num_hops));
        auto pages = stream.arm_unicast_write(page_size);  // invariant per-page payload size
        auto counter = stream.arm_inc(1);                  // invariant counting inc value

        // Both incs target the SAME sem_dir address, at two cores of the neighbour.
        const uint64_t neighbor_relay_sem = safe_get_noc_addr(relay_noc_x, relay_noc_y, sem_addr, 0);
        const uint64_t neighbor_reduce_sem = safe_get_noc_addr(reduce_noc_x, reduce_noc_y, sem_addr, 0);

        // Forward each block (seed first, then relays) one hop. The relay reader
        // pushes the same block order into cb_relay_pages, so a single FIFO drain
        // matches (T3: ring-modular indices; the line values never wrap).
        for (uint32_t k = 0; k < num_sends; ++k) {
            uint32_t c;
            if (k == 0) {
                c = my_chip_id;  // seed
            } else {
                c = (direction == 0) ? (my_chip_id + ring_size - k) % ring_size
                                     : (my_chip_id + k) % ring_size;  // relays
            }
            for (uint32_t p = 0; p < P; ++p) {
                cb_wait_front(cb_relay_pages, 1);
                const uint32_t l1 = get_read_ptr(cb_relay_pages);
                pages.write_page(l1, c * P + p, gather_buffer);
                noc_async_writes_flushed();  // R7: the page must be read before CB slot reuse
                cb_pop_front(cb_relay_pages, 1);
            }
            // T4: two in-order incs behind the block's pages on the SAME connection —
            // relay core (store-and-forward chain) + reduce core (overlap trigger).
            counter.inc(neighbor_relay_sem);
            counter.inc(neighbor_reduce_sem);
        }

        stream.close();  // drains (write + atomic barriers) then closes
    }
}
