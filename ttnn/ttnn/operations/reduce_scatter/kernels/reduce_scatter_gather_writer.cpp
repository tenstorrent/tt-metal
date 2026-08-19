// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — Phase A gather writer (BRISC): fabric egress through the
// safety-by-construction CCL helper (FabricStreamSender -> FabricStream ->
// UnicastWriteChannel / AtomicIncChannel). Structurally the proven all_reduce
// Phase-A writer. For chip i, direction d (0 = forward -> i+1, 1 = backward -> i-1):
//
//   * Forward writer: seed block i, then relay blocks i-1..0 -> neighbour i+1.
//   * Backward writer: seed block i, then relay blocks i+1..N-1 -> neighbour i-1.
//
// Every fabric write lands DIRECTLY into the downstream device's persistent
// gather_buffer at the block's canonical page range (through the LOCAL accessor
// base address routed one hop — uniform mesh addressing is load-bearing); one
// counting inc per landed block tells that device's reader the block arrived
// (in-order on the connection => the inc lands after the block's data). A line-end
// worker in its missing direction (my_num_targets == 0) opens no connection and
// returns — its RT arg list is empty, so ALL RT reads are guarded behind the
// early return.
//
// The op owns what the helper does NOT: the block relay order, the block page
// addressing (gb_page = c*P + p), and the counting atomic-inc target.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t direction = get_compile_time_arg_val(1);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_targets_fwd = get_compile_time_arg_val(3);
    constexpr uint32_t num_targets_bwd = get_compile_time_arg_val(4);
    constexpr uint32_t alignment = get_compile_time_arg_val(5);  // l1 alignment for fabric payloads
    constexpr auto gather_buffer_args = TensorAccessorArgs<6>();

    constexpr uint32_t my_num_targets = (direction == 0) ? num_targets_fwd : num_targets_bwd;
    constexpr uint32_t num_relay_blocks = (direction == 0) ? num_targets_bwd : num_targets_fwd;

    // Line end in this direction: no fabric egress, no connection opened.
    if constexpr (my_num_targets > 0) {
        uint32_t ai = 0;
        const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
        const uint32_t page_size = get_arg_val<uint32_t>(ai++);
        const uint32_t num_hops = get_arg_val<uint32_t>(ai++);
        const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t target_noc_x = get_arg_val<uint32_t>(ai++);
        const uint32_t target_noc_y = get_arg_val<uint32_t>(ai++);

        // Fabric connection arg block (laid out by the host's append-fabric-rt-args
        // idiom); its leading has_forward flag also encodes the send direction.
        size_t conn_arg_idx = ai;
        const bool dst_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
        FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

        const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
        const uint32_t P = pages_per_shard;

        auto stream = sender.open(unicast_route(num_hops));
        auto writer = stream.arm_unicast_write(page_size);  // invariant per-page payload size
        auto counter = stream.arm_inc(1);                   // invariant counting inc value

        const uint64_t neighbor_sem = safe_get_noc_addr(target_noc_x, target_noc_y, counting_sem_addr, 0);

        // Forward each block (seed first, then relays) one hop. The reader pushes
        // the same block order into cb_relay_pages, so a single FIFO drain matches.
        for (uint32_t k = 0; k <= num_relay_blocks; ++k) {
            uint32_t c;
            if (k == 0) {
                c = my_chip_id;  // seed
            } else {
                c = (direction == 0) ? (my_chip_id - k) : (my_chip_id + k);  // relays
            }
            for (uint32_t p = 0; p < P; ++p) {
                cb_wait_front(cb_relay_pages, 1);
                const uint32_t l1 = get_read_ptr(cb_relay_pages);
                writer.write_page(l1, c * P + p, gather_buffer);
                noc_async_writes_flushed();  // ensure the page was read before CB slot reuse
                cb_pop_front(cb_relay_pages, 1);
            }
            // In-order on the connection: this inc lands after the block's data.
            counter.inc(neighbor_sem);
        }

        stream.close();  // drains (write + atomic barriers) then closes
    }
}
