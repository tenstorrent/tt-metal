// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — per-direction reader (NCRISC).
//
// Pure data movement (no compute). One source file drives both the forward
// (direction=0, flow toward chip i+1) and backward (direction=1, flow toward
// chip i-1) worker cores; the direction CT arg selects the behaviour.
//
// Roles (chip id i, ring size N, concat along gather_dim):
//   out_page(chip c, local page p) = gather_out_page(c, p, dim_j, inner_stride, N)
//   (Refinement 2: strided concat addressing. For gather_dim=0 this reduces to
//    c*pages_per_shard + p — the page-contiguous case. See gather_out_page.)
//
//   * Self-copy (forward reader ONLY, every device): read this device's own
//     input shard and write it verbatim into its OWN output block i (local NoC).
//   * Seed (if this direction forwards, my_num_targets>0): stage this device's
//     input shard into cb_relay_pages for the writer to fabric-forward one hop.
//   * Relay / store-and-forward: for each block that arrives into local output
//     DRAM from the upstream neighbour, wait on the counting semaphore (the
//     SENDING half — the upstream writer's atomic-inc — lands the data first via
//     fabric in-order delivery), read the landed block BACK out of local output
//     DRAM into cb_relay_pages, and the writer forwards it one more hop. There is
//     no FabricStreamReceiver: the receive ingress is this local noc_async_read.
//   * Line-end (my_num_targets==0): pure receiver — just wait on the counting
//     semaphore to confirm all upstream blocks landed before the op completes.
//   * Cache-reuse re-arm: reset the counting semaphore after the last wait.

#include "api/dataflow/dataflow_api.h"

// Whole-page concat-by-gather_dim addressing (Refinement 2). `dim_j` = the
// gathered axis's size in the shard's page grid (TILE: [B,C,Ht,Wt]; RM:
// [B,C,H]); `inner_stride` = product of page-grid dims INNER to the gathered
// axis. Reduces to c*P+p for gather_dim=0 (dim_j=B_pages, inner_stride=P/B).
// Verified against torch.cat in test_all_gather_debug.py. (Duplicated verbatim
// in the writer — kept inline to avoid a JIT include-path dependency.)
inline uint32_t gather_out_page(uint32_t c, uint32_t p, uint32_t dim_j, uint32_t inner_stride, uint32_t ring_size) {
    const uint32_t block = dim_j * inner_stride;
    const uint32_t high = p / block;
    const uint32_t rem = p % block;
    const uint32_t mid = rem / inner_stride;
    const uint32_t low = rem % inner_stride;
    return high * (ring_size * block) + (c * dim_j + mid) * inner_stride + low;
}

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t direction = get_compile_time_arg_val(1);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_targets_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t num_targets_bwd = get_compile_time_arg_val(5);
    constexpr uint32_t cb_self_copy = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Devices this direction forwards to; and the number of relay blocks that
    // arrive from the opposite-side neighbour (store-and-forward read-backs).
    constexpr uint32_t my_num_targets = (direction == 0) ? num_targets_fwd : num_targets_bwd;
    constexpr uint32_t num_relay_blocks = (direction == 0) ? num_targets_bwd : num_targets_fwd;

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t dim_j = get_arg_val<uint32_t>(ai++);         // gathered-axis page size
    const uint32_t inner_stride = get_arg_val<uint32_t>(ai++);  // pages inner to gathered axis

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto output = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t P = pages_per_shard;
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counting_sem_addr);

    // 1. Self-copy: own input shard -> own output block i (forward reader, always).
    if constexpr (direction == 0) {
        cb_reserve_back(cb_self_copy, 1);
        const uint32_t scratch = get_write_ptr(cb_self_copy);
        for (uint32_t p = 0; p < P; ++p) {
            noc_async_read(input.get_noc_addr(p), scratch, page_size);
            noc_async_read_barrier();
            const uint32_t out_p = gather_out_page(my_chip_id, p, dim_j, inner_stride, ring_size);
            noc_async_write(scratch, output.get_noc_addr(out_p), page_size);
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
        // 3. Relay: read upstream-arrived blocks back out of local output DRAM.
        uint32_t running = 0;
        for (uint32_t k = 0; k < num_relay_blocks; ++k) {
            const uint32_t c = (direction == 0) ? (my_chip_id - 1 - k) : (my_chip_id + 1 + k);
            running += 1;
            noc_semaphore_wait_min(sem_ptr, running);
            for (uint32_t p = 0; p < P; ++p) {
                cb_reserve_back(cb_relay_pages, 1);
                const uint32_t l1 = get_write_ptr(cb_relay_pages);
                const uint32_t out_p = gather_out_page(c, p, dim_j, inner_stride, ring_size);
                noc_async_read(output.get_noc_addr(out_p), l1, page_size);
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
