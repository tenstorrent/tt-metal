// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — reader (NCRISC), shared across both phases (selected by CT arg 0).
//
// PHASE A (gather, phase==0): line store-and-forward gather into gather_buffer,
//   structurally the all_gather gather_dim=0 reader. One source drives both the
//   forward (direction=0, flow toward chip i+1) and backward (direction=1, flow
//   toward chip i-1) worker cores.
//     gb_page(block c, local page p) = c * pages_per_shard + p
//   * Self-copy (forward reader ONLY, every device): read this device's own input
//     shard and write it verbatim into its OWN gather_buffer block i (local NoC).
//   * Seed (if this direction forwards): stage the input shard into cb_relay_pages
//     for the writer to fabric-forward one hop.
//   * Relay / store-and-forward: for each block that lands in local gather_buffer
//     from the upstream neighbour, wait on the counting semaphore, read the block
//     BACK out of local gather_buffer into cb_relay_pages, and the writer forwards
//     it one more hop. There is no FabricStreamReceiver — the receive ingress is
//     this local noc_async_read.
//   * Line-end (my_num_targets==0): pure receiver — just wait on the counting
//     semaphore. Cache-reuse re-arm: reset the counting semaphore after the wait.
//
// PHASE B (reduce, phase==1): for each owned output-tile position i, read the N
//   gather blocks' tile i (gather_buffer[c*P + i], c=0..N-1) into cb_gathered_shards
//   in block order — the local compute kernel sums them. Pure local NoC reads.
//
// Uniform CT superset keeps the discarded if-constexpr branch in-bounds:
//   [0]=phase, [1..7]=scalars, then TWO TensorAccessorArgs (input+gather_buffer for
//   gather; gather_buffer+output for reduce, the 2nd unused here).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PHASE_GATHER = 0;
constexpr uint32_t PHASE_REDUCE = 1;

void kernel_main() {
    constexpr uint32_t phase = get_compile_time_arg_val(0);

    if constexpr (phase == PHASE_GATHER) {
        // ---------------------------------------------------------------------
        // Phase A — line store-and-forward gather.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(1);
        constexpr uint32_t cb_self_copy = get_compile_time_arg_val(2);
        constexpr uint32_t direction = get_compile_time_arg_val(3);  // 0 = forward, 1 = backward
        constexpr uint32_t my_chip_id = get_compile_time_arg_val(4);
        constexpr uint32_t ring_size = get_compile_time_arg_val(5);
        constexpr uint32_t num_targets_fwd = get_compile_time_arg_val(6);
        constexpr uint32_t num_targets_bwd = get_compile_time_arg_val(7);
        constexpr auto input_args = TensorAccessorArgs<8>();
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
            cb_reserve_back(cb_self_copy, 1);
            const uint32_t scratch = get_write_ptr(cb_self_copy);
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
    } else {
        // ---------------------------------------------------------------------
        // Phase B — read the N shard tiles for each owned output-tile position.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_gathered_shards = get_compile_time_arg_val(1);
        constexpr uint32_t num_devices = get_compile_time_arg_val(2);      // N
        constexpr uint32_t pages_per_shard = get_compile_time_arg_val(3);  // P
        constexpr auto gather_buffer_args = TensorAccessorArgs<8>();
        // Second accessor (output) is unused by the reduce reader; declared only to
        // keep the discarded gather branch's second-accessor offset in-bounds.
        [[maybe_unused]] constexpr auto unused_output_args =
            TensorAccessorArgs<gather_buffer_args.next_compile_time_args_offset()>();

        uint32_t ai = 0;
        const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t page_size = get_arg_val<uint32_t>(ai++);
        const uint32_t start_tile = get_arg_val<uint32_t>(ai++);
        const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);

        const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
        const uint32_t P = pages_per_shard;

        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint32_t i = start_tile + t;
            cb_reserve_back(cb_gathered_shards, num_devices);
            uint32_t l1 = get_write_ptr(cb_gathered_shards);
            for (uint32_t c = 0; c < num_devices; ++c) {
                noc_async_read(gather_buffer.get_noc_addr(c * P + i), l1, page_size);
                l1 += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gathered_shards, num_devices);
        }
    }
}
