// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — ring dataflow WRITER (BRISC). ONE source, run on both cores of a
// device (core_fwd = (0,0), core_bwd = (0,1)); `direction` (CT arg) selects the
// forward vs backward channel. Pure data movement (no compute).
//
// Per slice fed by the paired reader through cb_relay, this:
//   * (core_fwd, own slice only) LOCALLY writes S_d into this device's own output
//     slot d — a same-chip noc_async_write (NOT a fabric hop, so raw, not a helper);
//   * (if a neighbour exists) FABRIC-writes the slice one hop to the neighbour's
//     OUTPUT tensor at the concat offset (writer.write_page), then a single counting
//     arm_inc so the neighbour's reader learns "slice landed" (data-before-inc, same
//     1-hop fabric route as the payload).
//
// Fabric egress is the safety-by-construction CCL helper (FabricStreamSender ->
// FabricStream -> UnicastWriteChannel / AtomicIncChannel / MulticastIncChannel).
// The op owns: the N-party barrier WAIT + reset (raw noc_semaphore_wait_min / set),
// the local own-slot write, and concat-by-gather_dim output addressing.
//
// Advisory deviation (logged): per-tile UNICAST fabric writes (write_page) with one
// counting inc per whole slice, instead of coalescing up to num_tiles_per_packet
// tiles into a scatter packet. Identical output (counting is per-slice regardless),
// and it sidesteps the scatter CB-contiguity / partial-packet edge cases. A future
// perf refinement can reintroduce scatter coalescing.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t ring_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_relay = get_compile_time_arg_val(2);
    constexpr uint32_t direction = get_compile_time_arg_val(3);  // 0 = forward, 1 = backward
    constexpr uint32_t pages_per_shard = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t has_neighbor = get_compile_time_arg_val(6);        // sends over fabric in this direction
    constexpr uint32_t num_consume_slices = get_compile_time_arg_val(7);  // slices popped from cb_relay
    constexpr uint32_t is_fwd_core = get_compile_time_arg_val(8);         // owns the barrier + own-slot local write
    constexpr uint32_t alignment = get_compile_time_arg_val(9);
    constexpr uint32_t barrier_range = get_compile_time_arg_val(10);  // line-multicast range (hops) for the barrier
    constexpr auto output_args = TensorAccessorArgs<11>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t barrier_sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(2);  // neighbour's fwd/bwd counting sem
    const uint32_t barrier_target_x = get_arg_val<uint32_t>(3);   // fwd core (0,0) NOC x on peers
    const uint32_t barrier_target_y = get_arg_val<uint32_t>(4);
    const uint32_t counting_target_x = get_arg_val<uint32_t>(5);  // this direction's core NOC x on the neighbour
    const uint32_t counting_target_y = get_arg_val<uint32_t>(6);

    const auto output_acc = TensorAccessor(output_args, output_addr, page_size);
    auto* barrier_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);

    if constexpr (has_neighbor) {
        // ----- fabric connection (built from the append_ccl_fabric_rt_args block at idx 7) -----
        size_t conn_arg_idx = 7;
        const bool is_forward = get_arg_val<uint32_t>(conn_arg_idx);  // leading has_forward flag
        FabricStreamSender<> sender(conn_arg_idx, is_forward, alignment);
        auto stream = sender.open(unicast_route(1));  // every unicast write + inc is one hop

        // ----- N-party readiness barrier: multicast our inc to all peers in this direction -----
        // Block-scope the multicast channel so its header is freed before the counting arm_inc.
        {
            ccl_routing_utils::line_multicast_route_info_t mroute{};
            mroute.start_distance_in_hops = 1;
            mroute.range_hops = static_cast<uint16_t>(barrier_range);
            auto barrier = stream.arm_multicast_inc(mroute, 1);
            // Both directions target the FWD core (0,0), the single barrier owner on each peer.
            barrier.multicast_inc(safe_get_noc_addr(barrier_target_x, barrier_target_y, barrier_sem_addr, 0));
        }

        // Only the fwd core waits/resets the barrier (single owner -> no cross-core reset race).
        if constexpr (is_fwd_core) {
            noc_semaphore_wait_min(barrier_sem, ring_size - 1);
            noc_semaphore_set(barrier_sem, 0);
        }

        // ----- send own slice + relays over the fabric, one counting inc per slice -----
        auto writer = stream.arm_unicast_write(page_size);
        auto counter = stream.arm_inc(1);
        const uint64_t counting_noc = safe_get_noc_addr(counting_target_x, counting_target_y, counting_sem_addr, 0);

        for (uint32_t s = 0; s < num_consume_slices; ++s) {
            // Forward sends S_d, S_{d-1}, ... (origin j = d - s); backward sends S_d, S_{d+1}, ...
            const uint32_t j = (direction == 0) ? (ring_index - s) : (ring_index + s);
            const uint32_t slot_base = j * pages_per_shard;  // gather_dim=0: slice j -> output slot j (contiguous)
            const bool own_slice = (s == 0);

            for (uint32_t t = 0; t < pages_per_shard; ++t) {
                cb_wait_front(cb_relay, 1);
                const uint32_t l1_read_addr = get_read_ptr(cb_relay);
                const uint32_t tile_id = slot_base + t;

                // core_fwd places its OWN slice locally in its own output slot d (same chip).
                if constexpr (is_fwd_core) {
                    if (own_slice) {
                        noc_async_write(l1_read_addr, output_acc.get_noc_addr(tile_id), page_size);
                    }
                }
                // Fabric: one hop to the neighbour's output tile at the same concat offset.
                writer.write_page(l1_read_addr, tile_id, output_acc);

                noc_async_writes_flushed();  // both NoC sources drained -> l1 tile safe to recycle
                cb_pop_front(cb_relay, 1);
            }
            // Counting inc AFTER the slice's writes (fabric FIFO => data lands before the inc).
            counter.inc(counting_noc);
        }

        stream.close();             // drains fabric writes + incs, then closes (RAII backstop too)
        noc_async_write_barrier();  // ensure the local own-slot writes are acked
    } else {
        // ----- end device in this direction: no fabric egress -----
        if constexpr (is_fwd_core) {
            // core_fwd (d = N-1) is still the barrier owner and still writes its own slot locally.
            noc_semaphore_wait_min(barrier_sem, ring_size - 1);
            noc_semaphore_set(barrier_sem, 0);

            for (uint32_t s = 0; s < num_consume_slices; ++s) {  // == 1 (own slice)
                const uint32_t slot_base = ring_index * pages_per_shard;
                for (uint32_t t = 0; t < pages_per_shard; ++t) {
                    cb_wait_front(cb_relay, 1);
                    const uint32_t l1_read_addr = get_read_ptr(cb_relay);
                    noc_async_write(l1_read_addr, output_acc.get_noc_addr(slot_base + t), page_size);
                    noc_async_writes_flushed();
                    cb_pop_front(cb_relay, 1);
                }
            }
            noc_async_write_barrier();
        }
        // core_bwd (d = 0): num_consume_slices == 0, nothing to send — no-op.
    }
}
