// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter_average — relay reader (NCRISC), cores (0,0) [forward] and (0,1)
// [backward], direction selected by CT arg.
//
// One half of the line store-and-forward gather (op_design.md "Dataflow Strategy"):
//   * Seed (num_sends > 0): stage this device's own input shard, page by page, into
//     cb_relay_pages for the relay writer to fabric-forward one hop.
//   * Store-and-forward: for each of num_arrivals blocks that the upstream neighbour
//     lands in the LOCAL gather_buffer, wait on this core's counting semaphore
//     (sem >= k+1 implies the block's data is complete — the sender's inc is in-order
//     behind the pages on the same fabric connection, T4/R8); the first
//     num_sends - 1 arrivals are read BACK out of gather_buffer into cb_relay_pages
//     for the writer to forward one more hop. Remaining arrivals terminate here and
//     are awaited only (a line-end device relays nothing but still waits ALL its
//     arrivals so no in-flight inc can land after the re-arm, T6).
//   * Cache-reuse re-arm (R1): reset this core's semaphore counter to 0 after the
//     final wait — on every role, including pure receivers. A missing reset passes
//     the first call and hangs/corrupts the program-cache-hit second call.
//
// Unlike the reference reduce_scatter there is NO self-copy: the reduce reader takes
// this device's own contribution directly from the input tensor, so gather_buffer
// block my_chip_id is never written (op_design.md "Op-internal gather buffer").
//
// CT args: [cb_relay_pages, direction, my_chip_id, ring_size, num_sends,
//           num_arrivals] + input TensorAccessorArgs + gather TensorAccessorArgs
// RT args: [input_addr, gather_buffer_addr, pages_per_shard, page_size, sem_addr]

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(0);
    constexpr uint32_t direction = get_compile_time_arg_val(1);  // 0 = forward, 1 = backward
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    // Blocks this direction's writer sends (seed + relays; 0 = idle line end) /
    // blocks landing here from the upstream neighbour. Host-derived block-flow
    // table (Linear: relay-everything; Ring: short-way depth — fwd N/2, bwd
    // (N-1)/2, where the last arrival terminates here and only the first
    // num_sends-1 are relayed onward).
    constexpr uint32_t num_sends = get_compile_time_arg_val(4);
    constexpr uint32_t num_arrivals = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto gather_buffer_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Store-and-forward invariant: relayed blocks are a PREFIX of arrivals.
    constexpr uint32_t num_relays = (num_sends > 0) ? num_sends - 1 : 0;
    static_assert(num_relays <= num_arrivals, "reduce_scatter_average: relayed blocks must be a prefix of arrivals");

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(ai++);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
    const uint32_t P = pages_per_shard;
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    // 1. Seed: stage the own input shard for the writer to forward one hop.
    if constexpr (num_sends > 0) {
        for (uint32_t p = 0; p < P; ++p) {
            cb_reserve_back(cb_relay_pages, 1);
            const uint32_t l1 = get_write_ptr(cb_relay_pages);
            noc_async_read(input.get_noc_addr(p), l1, page_size);
            noc_async_read_barrier();
            cb_push_back(cb_relay_pages, 1);
        }
    }

    // 2. Arrival waits + store-and-forward read-backs. Arrival k carries the shard of
    //    chip (my_chip_id -/+ (1 + k)) mod N — nearest-first chain order (T1/T2). The
    //    ring-modular form never wraps on the Linear line; on Ring it wraps at the
    //    (N-1) <-> 0 seam (Refinement 2) — the kernel is topology-agnostic.
    for (uint32_t k = 0; k < num_arrivals; ++k) {
        noc_semaphore_wait_min(sem_ptr, k + 1);
        if (k < num_relays) {
            const uint32_t c =
                (direction == 0) ? (my_chip_id + ring_size - 1 - k) % ring_size : (my_chip_id + 1 + k) % ring_size;
            for (uint32_t p = 0; p < P; ++p) {
                cb_reserve_back(cb_relay_pages, 1);
                const uint32_t l1 = get_write_ptr(cb_relay_pages);
                noc_async_read(gather_buffer.get_noc_addr(c * P + p), l1, page_size);
                noc_async_read_barrier();
                cb_push_back(cb_relay_pages, 1);
            }
        }
    }

    // 3. Cache-reuse re-arm (R1): after the final wait no fwd/bwd inc for this run can
    //    still be in flight to this core, so the reset cannot race a sender.
    noc_semaphore_set(sem_ptr, 0);
}
