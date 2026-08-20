// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — reader (NCRISC), shared across both phases (selected by CT arg 0).
//
// PHASE A (gather, phase==0): store-and-forward gather into gather_buffer over a
//   LINE or a RING (Refinement 1) — structurally the proven all_reduce Phase-A
//   reader, generalized to ring-modular block indices and per-direction
//   (num_sends, num_arrivals) counts the HOST derives from the topology:
//     Linear: fwd of device i sends 1+i blocks (seed + all line arrivals),
//             receives i;      bwd mirrored. Line ends send 0.
//     Ring:   fwd sends/receives fwd_depth = N/2 blocks; bwd sends/receives
//             bwd_depth = (N-1)/2 — each direction relays only its short-way
//             half of the ring, so every block lands EXACTLY once per device.
//   The store-and-forward invariant unifying both: the writer's sends are 1 seed
//   + (num_sends - 1) relays, drawn from the FIRST num_sends - 1 arrivals; any
//   remaining arrivals are awaited only (they terminate here).
//     gb_page(block c, local page p) = c * pages_per_shard + p
//   * Self-copy (forward reader ONLY, every device): read this device's own input
//     shard and write it verbatim into its OWN gather_buffer block i (local NoC),
//     bounced through the cb_self_copy scratch slot.
//   * Seed (if num_sends > 0): stage the input shard into cb_relay_pages for the
//     writer to fabric-forward one hop.
//   * Relay / store-and-forward: for each block that lands in local gather_buffer
//     from the upstream neighbour, wait on the counting semaphore; the first
//     num_sends-1 arrivals are read BACK out of local gather_buffer into
//     cb_relay_pages for the writer to forward one more hop. There is no
//     FabricStreamReceiver — the receive ingress is this local noc_async_read
//     (op-owned by the fabric helper's documented split).
//   * Cache-reuse re-arm: reset the counting semaphore after the LAST wait.
//
// PHASE B (reduce, phase==1): for each owned OUTPUT-tile position t in
//   [start, start+n), compute the slice tile id ONCE via the shared-schedule
//   SliceRowWalker (the ONE definition of slice addressing — the same type the
//   silicon-verified C++ reduce_scatter kernels use), then read the N gathered
//   tiles gather_buffer[c*P + id] for c = 0..N-1 in block order into ONE
//   cb_gathered_slices reservation of N pages; one barrier; push N. Block-major
//   order is load-bearing: sum_blocks reads block c at CB page c.
//
// Uniform CT superset keeps the discarded if-constexpr branch in-bounds:
//   [0]=phase, [1..7]=scalars, then TWO TensorAccessorArgs (input+gather_buffer for
//   gather; gather_buffer+output for reduce, the 2nd unused there).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;

constexpr uint32_t PHASE_GATHER = 0;
constexpr uint32_t PHASE_REDUCE = 1;

void kernel_main() {
    constexpr uint32_t phase = get_compile_time_arg_val(0);

    // Compile-time gate on the scatter dim (Phase B only). Guarded on phase because
    // a discarded if-constexpr branch in a non-template function is still fully
    // checked, and in Phase A scalar slot 4 aliases my_chip_id (which is 0 on chip 0).
    static_assert(
        phase == PHASE_GATHER || sched::is_supported_scatter_dim(get_compile_time_arg_val(4)),
        "reduce_scatter: unsupported scatter dim");

    if constexpr (phase == PHASE_GATHER) {
        // ---------------------------------------------------------------------
        // Phase A — line store-and-forward gather.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(1);
        constexpr uint32_t cb_self_copy = get_compile_time_arg_val(2);
        constexpr uint32_t direction = get_compile_time_arg_val(3);  // 0 = forward, 1 = backward
        constexpr uint32_t my_chip_id = get_compile_time_arg_val(4);
        constexpr uint32_t ring_size = get_compile_time_arg_val(5);
        // Per-direction, topology-derived by the HOST (see file banner): blocks this direction's
        // writer sends (seed + relays; 0 = idle direction) / blocks landing here from upstream.
        constexpr uint32_t num_sends = get_compile_time_arg_val(6);
        constexpr uint32_t num_arrivals = get_compile_time_arg_val(7);
        constexpr auto input_args = TensorAccessorArgs<8>();
        constexpr auto gather_buffer_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

        // Store-and-forward invariant (both topologies): relays are a PREFIX of arrivals.
        // (Guard on phase: a discarded if-constexpr branch is still fully checked, and in
        // Phase B these CT slots alias slice_run_len / slice_stride.)
        constexpr uint32_t num_relays = (num_sends > 0) ? num_sends - 1 : 0;
        static_assert(
            phase != PHASE_GATHER || num_relays <= num_arrivals,
            "reduce_scatter: relayed blocks must be a prefix of arrivals");

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
        //    cb_self_copy is reserve-only SCRATCH — never pushed/popped (intentional; see
        //    op_design.md §R9). One reservation, write-ptr reused per page.
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

        if constexpr (num_sends > 0) {
            // 2. Seed: stage own input shard for the writer to forward one hop.
            for (uint32_t p = 0; p < P; ++p) {
                cb_reserve_back(cb_relay_pages, 1);
                const uint32_t l1 = get_write_ptr(cb_relay_pages);
                noc_async_read(input.get_noc_addr(p), l1, page_size);
                noc_async_read_barrier();
                cb_push_back(cb_relay_pages, 1);
            }
        }

        // 3. Arrival waits + store-and-forward read-backs. Arrival k is block
        //    (my_chip_id -/+ (1 + k)) mod N — ring-modular; the line values never wrap,
        //    so this is exactly the pre-ring `my_chip_id -/+ (1 + k)` walk there. The
        //    counting wait guarantees the block DATA landed (the sending writer's inc is
        //    in-order on the fabric connection, after the block's pages). The first
        //    num_relays arrivals are read back for the writer to forward one more hop;
        //    the rest terminate here and are awaited only.
        uint32_t running = 0;
        for (uint32_t k = 0; k < num_arrivals; ++k) {
            running += 1;
            noc_semaphore_wait_min(sem_ptr, running);
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

        // 4. Cache-reuse re-arm: reset the counting semaphore after the LAST wait, on
        //    EVERY role including pure receivers (op_design.md §R1 — a missing re-arm
        //    passes the first call and hangs/corrupts the program-cache-hit second call).
        noc_semaphore_set(sem_ptr, 0);
    } else {
        // ---------------------------------------------------------------------
        // Phase B — read the N gathered tiles at each owned output position's
        // SLICE tile id (SliceRowWalker), block-major into cb_gathered_slices.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_gathered_slices = get_compile_time_arg_val(1);
        constexpr uint32_t num_devices = get_compile_time_arg_val(2);           // N
        constexpr uint32_t pages_per_shard = get_compile_time_arg_val(3);       // P (input shard pages)
        [[maybe_unused]] constexpr uint32_t dim = get_compile_time_arg_val(4);  // gated by static_assert above
        constexpr uint32_t slice_base = get_compile_time_arg_val(5);            // first tile id of device i's slice
        constexpr uint32_t slice_run_len = get_compile_time_arg_val(6);         // contiguous tile run inside the slice
        constexpr uint32_t slice_stride = get_compile_time_arg_val(7);          // tile-id jump between runs
        constexpr auto gather_buffer_args = TensorAccessorArgs<8>();
        // Second accessor (output) is unused by the reduce reader; declared only to
        // keep the discarded gather branch's second-accessor offset in-bounds.
        [[maybe_unused]] constexpr auto unused_output_args =
            TensorAccessorArgs<gather_buffer_args.next_compile_time_args_offset()>();

        uint32_t ai = 0;
        const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t page_size = get_arg_val<uint32_t>(ai++);
        const uint32_t start_tile = get_arg_val<uint32_t>(ai++);  // first owned OUTPUT position
        const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);   // owned OUTPUT positions

        const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
        const uint32_t P = pages_per_shard;

        // Seed the shared-schedule walker at this core's first owned position
        // (op_design.md "Slice addressing" — the seed formula is the only nontrivial
        // piece; taken from the design, not re-derived).
        sched::SliceRowWalker walker(slice_run_len, slice_stride);
        walker.set_base(slice_base);
        walker.reset_offsets(start_tile % slice_run_len, (start_tile / slice_run_len) * slice_stride);

        for (uint32_t t = 0; t < num_tiles; ++t) {
            // ONE next() per output position — it returns AND advances (§R5); the id is
            // reused for all N block reads.
            const uint32_t id = walker.next();
            cb_reserve_back(cb_gathered_slices, num_devices);
            uint32_t l1 = get_write_ptr(cb_gathered_slices);
            for (uint32_t c = 0; c < num_devices; ++c) {
                noc_async_read(gather_buffer.get_noc_addr(c * P + id), l1, page_size);
                l1 += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gathered_slices, num_devices);
        }
    }
}
