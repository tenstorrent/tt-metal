// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — per-direction worker WRITER kernel (BRISC).
//
// One instance runs on the forward core (0,0) with direction=0 and one on the
// backward core (0,1) with direction=1 of every device on the 1-D line.
//
// Fabric egress uses the safety-by-construction CCL helper
// (FabricStreamSender -> FabricStream -> {MulticastIncChannel, UnicastWriteChannel,
// AtomicIncChannel}). The op OWNS (helper does not, by design):
//   * the local self-copy of this device's own block (intra-device noc_async_write),
//   * the store-and-forward concat addressing (out_page = c*pages_per_shard + p),
//   * the WAITING half of the barrier + the barrier/counting semaphore resets.
//
// Phases: (1) N-party startup barrier via multicast atomic-inc; (3) seed the own
// block (self-copy on the forward core, always; fabric-forward if it has targets);
// (5) relay upstream-arrived blocks one more hop, one counting inc per block; (6)
// drain + close.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t direction = get_compile_time_arg_val(0);  // 0 = forward, 1 = backward
    constexpr uint32_t ring_size = get_compile_time_arg_val(1);
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_shard = get_compile_time_arg_val(3);
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_targets_forward = get_compile_time_arg_val(6);
    constexpr uint32_t num_targets_backward = get_compile_time_arg_val(7);
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(8);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(9);
    constexpr auto output_args = TensorAccessorArgs<10>();

    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t barrier_sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t fwd_core_x = get_arg_val<uint32_t>(3);
    const uint32_t fwd_core_y = get_arg_val<uint32_t>(4);
    const uint32_t bwd_core_x = get_arg_val<uint32_t>(5);
    const uint32_t bwd_core_y = get_arg_val<uint32_t>(6);

    constexpr bool is_forward = (direction == 0);
    constexpr bool will_forward = is_forward ? (num_targets_forward > 0) : (num_targets_backward > 0);
    constexpr uint32_t num_relay = is_forward ? num_targets_backward : num_targets_forward;
    constexpr uint32_t mcast_range = is_forward ? num_targets_forward : num_targets_backward;

    const auto output_accessor = TensorAccessor(output_args, output_base_addr, page_size);

    volatile tt_l1_ptr uint32_t* barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);

    // The counting inc targets the SAME-role core on the immediate neighbour (uniform
    // grid -> identical virtual coords), routed one hop in this worker's direction.
    const uint64_t counting_target_noc = is_forward ? safe_get_noc_addr(fwd_core_x, fwd_core_y, counting_sem_addr, 0)
                                                    : safe_get_noc_addr(bwd_core_x, bwd_core_y, counting_sem_addr, 0);

    if constexpr (will_forward) {
        // The fabric arg block (laid out by append_ccl_fabric_rt_args) begins at index
        // 7; its leading has_forward flag also encodes the send direction.
        size_t conn_arg_idx = 7;
        const bool sender_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
        FabricStreamSender<> sender(conn_arg_idx, sender_is_forward, l1_alignment);
        auto stream = sender.open(unicast_route(1));  // immediate neighbour (1 hop)

        // ---- Phase 1: N-party startup barrier (multicast atomic-inc) ----
        // Block-scope the MulticastIncChannel to the barrier phase (structural hygiene;
        // its header is independent of the unicast channels below).
        {
            ccl_routing_utils::line_multicast_route_info_t mcast_route{};
            mcast_route.start_distance_in_hops = 1;
            mcast_route.range_hops = mcast_range;
            mcast_route.e_num_hops = 0;
            mcast_route.w_num_hops = 0;
            mcast_route.n_num_hops = 0;
            mcast_route.s_num_hops = 0;
            auto barrier = stream.arm_multicast_inc(mcast_route, 1);
            // Hit BOTH cores of every reachable peer so each core receives ring_size-1.
            barrier.multicast_inc(safe_get_noc_addr(fwd_core_x, fwd_core_y, barrier_sem_addr, 0));
            barrier.multicast_inc(safe_get_noc_addr(bwd_core_x, bwd_core_y, barrier_sem_addr, 0));
        }
        noc_semaphore_wait_min(barrier_sem_ptr, ring_size - 1);
        noc_semaphore_set(barrier_sem_ptr, 0);

        // Arm the payload + counting channels (each draws its own pooled header, all
        // reusing the stream's route bound at open()).
        auto writer = stream.arm_unicast_write(aligned_page_size);
        auto counter = stream.arm_inc(1);

        // ---- Phase 3: seed (own block `my_chip_id`) ----
        {
            const uint32_t base = my_chip_id * pages_per_shard;
            for (uint32_t p = 0; p < pages_per_shard; ++p) {
                cb_wait_front(cb_relay_pages, 1);
                const uint32_t src = get_read_ptr(cb_relay_pages);
                const uint32_t out_page = base + p;
                if constexpr (is_forward) {
                    // Intra-device self-copy of the own shard into own output block.
                    noc_async_write(src, output_accessor.get_noc_addr(out_page), aligned_page_size);
                }
                // Fabric-forward the seed to the neighbour's identical output page.
                writer.write_page(src, out_page, output_accessor);
                if constexpr (is_forward) {
                    noc_async_write_barrier();  // local self-copy done before slot reuse
                }
                cb_pop_front(cb_relay_pages, 1);
            }
            counter.inc(counting_target_noc);  // one counting inc per block
        }

        // ---- Phase 5: relay upstream-arrived blocks one more hop ----
        for (uint32_t k = 0; k < num_relay; ++k) {
            const uint32_t c = is_forward ? (my_chip_id - 1 - k) : (my_chip_id + 1 + k);
            const uint32_t base = c * pages_per_shard;
            for (uint32_t p = 0; p < pages_per_shard; ++p) {
                cb_wait_front(cb_relay_pages, 1);
                const uint32_t src = get_read_ptr(cb_relay_pages);
                writer.write_page(src, base + p, output_accessor);
                cb_pop_front(cb_relay_pages, 1);
            }
            counter.inc(counting_target_noc);
        }

        // ---- Phase 6: teardown ----
        stream.close();             // drains fabric writes + incs, then closes
        noc_async_write_barrier();  // backstop for any local self-copies
    } else {
        // Line-end worker in its missing direction: no fabric egress. It still joins
        // the barrier (receives ring_size-1 incs from the other devices) and, on the
        // forward core, still self-copies its own block.
        noc_semaphore_wait_min(barrier_sem_ptr, ring_size - 1);
        noc_semaphore_set(barrier_sem_ptr, 0);

        if constexpr (is_forward) {
            const uint32_t base = my_chip_id * pages_per_shard;
            for (uint32_t p = 0; p < pages_per_shard; ++p) {
                cb_wait_front(cb_relay_pages, 1);
                const uint32_t src = get_read_ptr(cb_relay_pages);
                noc_async_write(src, output_accessor.get_noc_addr(base + p), aligned_page_size);
                noc_async_write_barrier();
                cb_pop_front(cb_relay_pages, 1);
            }
        }
    }
}
