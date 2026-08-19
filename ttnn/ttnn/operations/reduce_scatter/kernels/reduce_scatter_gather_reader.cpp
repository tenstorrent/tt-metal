// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — Phase A gather reader (NCRISC), one source for both direction
// cores (direction = CT arg): line store-and-forward gather into gather_buffer.
// Structurally the proven all_reduce Phase-A reader.
//
//   gb_page(block c, local page p) = c * pages_per_shard + p
//
//   * Self-copy (forward reader ONLY, every device): read this device's own input
//     shard and write it verbatim into its OWN gather_buffer block i (local NoC).
//     Uses cb_self_copy_scratch as reserve-only scratch — NEVER pushed/popped
//     (proven all_reduce idiom; do not "fix" into a push/pop CB).
//   * Seed (if this direction forwards): stage the input shard into cb_relay_pages
//     for the writer to fabric-forward one hop.
//   * Relay / store-and-forward: for each block that lands in local gather_buffer
//     from the upstream neighbour, wait on the counting semaphore, read the block
//     BACK out of local gather_buffer into cb_relay_pages, and the writer forwards
//     it one more hop. There is no FabricStreamReceiver — the receive ingress is
//     this local noc_async_read (op-owned per the ccl_helpers_dataflow.hpp banner).
//   * Line-end (my_num_targets == 0): pure receiver — just wait on the counting
//     semaphore.
//   * Cache-reuse re-arm: reset the counting semaphore to 0 AFTER the last wait
//     (RECEIVER resets after its wait). Missing this: first call green, second hangs.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_self_copy_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t direction = get_compile_time_arg_val(2);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_targets_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t num_targets_bwd = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto gather_buffer_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Devices this direction forwards to; and the number of relay blocks that
    // arrive from the opposite-side neighbour (store-and-forward read-backs).
    constexpr uint32_t my_num_targets = (direction == 0) ? num_targets_fwd : num_targets_bwd;
    constexpr uint32_t num_relay_blocks = (direction == 0) ? num_targets_bwd : num_targets_fwd;

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
    const uint32_t P = pages_per_shard;
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counting_sem_addr);

    // 1. Self-copy: own input shard -> own gather_buffer block i (forward reader, always).
    if constexpr (direction == 0) {
        cb_reserve_back(cb_self_copy_scratch, 1);
        const uint32_t scratch = get_write_ptr(cb_self_copy_scratch);
        for (uint32_t p = 0; p < P; ++p) {
            noc_async_read(input.get_noc_addr(p), scratch, page_size);
            noc_async_read_barrier();
            noc_async_write(scratch, gather_buffer.get_noc_addr(my_chip_id * P + p), page_size);
            noc_async_write_barrier();
        }
    }

    if constexpr (my_num_targets > 0) {
        // 2. Seed: stage own input shard for the writer to forward one hop.
        for (uint32_t p = 0; p < P; ++p) {
            cb_reserve_back(cb_relay_pages, 1);
            const uint32_t l1 = get_write_ptr(cb_relay_pages);
            noc_async_read(input.get_noc_addr(p), l1, page_size);
            noc_async_read_barrier();
            cb_push_back(cb_relay_pages, 1);
        }
        // 3. Relay: read upstream-arrived blocks back out of local gather_buffer.
        uint32_t running = 0;
        for (uint32_t k = 0; k < num_relay_blocks; ++k) {
            const uint32_t c = (direction == 0) ? (my_chip_id - 1 - k) : (my_chip_id + 1 + k);
            running += 1;
            noc_semaphore_wait_min(sem_ptr, running);
            for (uint32_t p = 0; p < P; ++p) {
                cb_reserve_back(cb_relay_pages, 1);
                const uint32_t l1 = get_write_ptr(cb_relay_pages);
                noc_async_read(gather_buffer.get_noc_addr(c * P + p), l1, page_size);
                noc_async_read_barrier();
                cb_push_back(cb_relay_pages, 1);
            }
        }
    } else {
        // Line end in this direction: pure receiver — confirm all blocks landed.
        if constexpr (num_relay_blocks > 0) {
            noc_semaphore_wait_min(sem_ptr, num_relay_blocks);
        }
    }

    // 4. Cache-reuse re-arm: reset the counting semaphore after the last wait.
    noc_semaphore_set(sem_ptr, 0);
}
