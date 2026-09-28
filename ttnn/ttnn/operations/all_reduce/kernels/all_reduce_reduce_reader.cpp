// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — reduce reader (NCRISC), core (0,2).
//
// Feeds the compute kernel N contributions of P tiles each, in ARRIVAL order, as
// g-tile granules through cb_contributions:
//   * contribution 0: this device's OWN shard, read directly from the input tensor —
//     no dependence on anything remote, so compute pass 0 starts immediately (T7).
//   * contributions 1..N-1: each remote shard the moment its double-inc lands in this
//     core's sem_fwd / sem_bwd counter — a TWO-WAY POLL (volatile reads +
//     invalidate_l1_cache, the same spin noc_semaphore_wait_min uses) so whichever
//     direction lands first is consumed first, preserving the
//     accumulate-overlaps-flight property. A single-counter noc_semaphore_wait_min
//     would serialize the directions (op_design.md "Arrival poll").
//   Fwd arrival a carries the shard of chip (i - (1+a)) mod N; bwd arrival b that of
//   chip (i + (1+b)) mod N (T1/T2, nearest-first).
//
// Every contribution is streamed in the IDENTICAL dense page order 0..P-1 of its
// block (base 0 for the input tensor, src*P for gather blocks) — the full-shard walk
// is the identity, which keeps add_tiles positionally aligned across passes (R11)
// AND makes the dense writer valid (tile t drained = output page t).
//
// Cache-reuse re-arm (R1): reset BOTH sem counters after all arrivals are consumed.
//
// CT args: [cb_contributions, my_chip_id, ring_size, fwd_arrivals, bwd_arrivals,
//           P, g] + input TensorAccessorArgs + gather TensorAccessorArgs
// RT args: [input_addr, gather_buffer_addr, page_size, sem_fwd_addr, sem_bwd_addr]

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_contributions = get_compile_time_arg_val(0);
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t fwd_arrivals = get_compile_time_arg_val(3);
    constexpr uint32_t bwd_arrivals = get_compile_time_arg_val(4);
    constexpr uint32_t P = get_compile_time_arg_val(5);  // tiles per shard = output tiles
    constexpr uint32_t g = get_compile_time_arg_val(6);  // CB granule (divides P)
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto gather_buffer_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    static_assert(g > 0 && P % g == 0, "all_reduce: granule must divide the shard tile count (R5)");
    static_assert(fwd_arrivals + bwd_arrivals + 1 == ring_size, "all_reduce: arrivals + own contribution must equal N");

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_fwd_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_bwd_addr = get_arg_val<uint32_t>(ai++);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
    auto sem_fwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_fwd_addr);
    auto sem_bwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_bwd_addr);

    // One contribution = P dense tiles (pages base..base+P-1), streamed as P/g
    // whole g-granules — no tail exists (g divides P by host construction, R5/R6).
    auto push_contribution = [&](const auto& src, uint32_t base) {
        for (uint32_t chunk = 0; chunk < P / g; ++chunk) {
            cb_reserve_back(cb_contributions, g);
            uint32_t l1 = get_write_ptr(cb_contributions);
            for (uint32_t t = 0; t < g; ++t) {
                noc_async_read(src.get_noc_addr(base + chunk * g + t), l1, page_size);
                l1 += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_contributions, g);
        }
    };

    // 1. Own contribution first — straight from the input tensor (gather_buffer block
    //    my_chip_id is never written; there is no serialized self-copy).
    push_contribution(input, 0);

    // 2. Remote contributions in ARRIVAL order: two-way monotonic-counter poll.
    //    sem_dir > consumed  <=>  at least one unconsumed arrival in that direction,
    //    and its data is fully landed (the inc is in-order behind the pages, R8).
    uint32_t fwd_consumed = 0;
    uint32_t bwd_consumed = 0;
    constexpr uint32_t total_arrivals = fwd_arrivals + bwd_arrivals;
    for (uint32_t done = 0; done < total_arrivals;) {
        invalidate_l1_cache();
        if (fwd_consumed < fwd_arrivals && *sem_fwd_ptr > fwd_consumed) {
            const uint32_t src_chip = (my_chip_id + ring_size - 1 - fwd_consumed) % ring_size;
            push_contribution(gather_buffer, src_chip * P);
            ++fwd_consumed;
            ++done;
        } else if (bwd_consumed < bwd_arrivals && *sem_bwd_ptr > bwd_consumed) {
            const uint32_t src_chip = (my_chip_id + 1 + bwd_consumed) % ring_size;
            push_contribution(gather_buffer, src_chip * P);
            ++bwd_consumed;
            ++done;
        }
    }

    // 3. Cache-reuse re-arm (R1): all fwd_arrivals + bwd_arrivals incs have been
    //    OBSERVED, so none can still be in flight — the reset cannot race a sender.
    noc_semaphore_set(sem_fwd_ptr, 0);
    noc_semaphore_set(sem_bwd_ptr, 0);
}
