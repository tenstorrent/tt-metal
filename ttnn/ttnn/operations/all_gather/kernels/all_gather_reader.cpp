// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — per-direction worker READER kernel (NCRISC).
//
// One instance runs on the forward core (0,0) with direction=0 and one on the
// backward core (0,1) with direction=1 of every device on the 1-D line. It is
// PURE DATA MOVEMENT (no fabric, no compute):
//
//   1. Seed: read this device's own input shard (block `my_chip_id`) into
//      cb_relay_pages, so the writer can self-copy it (forward) and fabric-forward
//      it. The forward reader ALWAYS seeds (the forward writer self-copies even at
//      the line end); the backward reader seeds only when it will forward.
//   2. Relay (store-and-forward INGRESS, op-owned per the helper banner — there is
//      no FabricStreamReceiver): for each block that upstream fabric-wrote into this
//      device's persistent output DRAM, wait on the counting semaphore, then read
//      the landed block back OUT of local output DRAM into cb_relay_pages so the
//      writer can forward it one more hop. Only done when this worker forwards.
//   3. Drain + re-arm: wait for all counting incs this device receives, then reset
//      the counting semaphore to 0 (cache-reuse re-arm — a RECEIVER resets after its
//      wait; ccl_helpers_dataflow.hpp:75-77).

#include "api/dataflow/dataflow_api.h"

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

    constexpr auto input_args = TensorAccessorArgs<9>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, input_base_addr, page_size);
    const auto output_accessor = TensorAccessor(output_args, output_base_addr, page_size);

    volatile tt_l1_ptr uint32_t* counting_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counting_sem_addr);

    constexpr bool is_forward = (direction == 0);
    // This worker forwards over fabric iff it has targets in its direction.
    constexpr bool will_forward = is_forward ? (num_targets_forward > 0) : (num_targets_backward > 0);
    // Number of upstream-arrived blocks we relay onward (= targets in the OPPOSITE
    // direction: the forward worker relays the left-arrived blocks, and vice-versa).
    constexpr uint32_t num_relay = is_forward ? num_targets_backward : num_targets_forward;
    // Total counting incs this device's same-direction core receives from its
    // immediate neighbour (forward: `my_chip_id`; backward: `ring_size-1-my_chip_id`).
    constexpr uint32_t recv_incs = is_forward ? my_chip_id : (ring_size - 1 - my_chip_id);

    // ---- Phase 2: seed ----
    // Forward: always (the forward writer self-copies its own block on every device).
    // Backward: only when it will forward (no self-copy on the backward core).
    constexpr bool read_seed = is_forward || will_forward;
    if constexpr (read_seed) {
        for (uint32_t p = 0; p < pages_per_shard; ++p) {
            cb_reserve_back(cb_relay_pages, 1);
            const uint32_t l1_addr = get_write_ptr(cb_relay_pages);
            noc_async_read(input_accessor.get_noc_addr(p), l1_addr, aligned_page_size);
            noc_async_read_barrier();
            cb_push_back(cb_relay_pages, 1);
        }
    }

    // ---- Phase 4: relay receive (read landed blocks back out of local output DRAM) ----
    if constexpr (will_forward) {
        for (uint32_t k = 0; k < num_relay; ++k) {
            // Block (k+1) has landed once the counting sem reaches k+1 (the upstream
            // writer issues one inc per block, with the fabric write flushed first).
            noc_semaphore_wait_min(counting_sem_ptr, k + 1);

            // Chip id of the block being relayed: forward relays the left blocks
            // (my_chip_id-1 .. 0), backward relays the right blocks (my_chip_id+1 .. N-1).
            const uint32_t c = is_forward ? (my_chip_id - 1 - k) : (my_chip_id + 1 + k);
            const uint32_t base = c * pages_per_shard;
            for (uint32_t p = 0; p < pages_per_shard; ++p) {
                cb_reserve_back(cb_relay_pages, 1);
                const uint32_t l1_addr = get_write_ptr(cb_relay_pages);
                noc_async_read(output_accessor.get_noc_addr(base + p), l1_addr, aligned_page_size);
                noc_async_read_barrier();
                cb_push_back(cb_relay_pages, 1);
            }
        }
    }

    // ---- Phase 6: drain + cache-reuse re-arm ----
    // Wait until every counting inc this core receives has landed, then reset to 0 so
    // a program-cache-hit re-run starts from a clean semaphore. A line-end worker that
    // does not forward still receives (and here drains) the incs from its neighbour.
    if constexpr (recv_incs > 0) {
        noc_semaphore_wait_min(counting_sem_ptr, recv_incs);
        noc_semaphore_set(counting_sem_ptr, 0);
    }
}
