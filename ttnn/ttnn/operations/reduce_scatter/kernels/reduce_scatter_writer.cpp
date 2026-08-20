// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — writer (BRISC), shared across both phases (selected by CT arg 0).
//
// PHASE A (gather, phase==0): fabric egress through the safety-by-construction CCL
//   helper (FabricStreamSender -> FabricStream -> UnicastWriteChannel /
//   AtomicIncChannel) — structurally the proven all_reduce Phase-A writer,
//   generalized (Refinement 1) to ring-modular block indices and a HOST-derived
//   per-direction send count. For chip i, direction d (0=forward -> (i+1)%N,
//   1=backward -> (i-1)%N), the writer sends num_sends blocks to its neighbour:
//     * k == 0: seed block i.
//     * k >= 1: relay block (i -/+ k) mod N (fwd/bwd) — the block the reader just
//       read back out of local gather_buffer. Linear feeds num_sends = 1 + all
//       line arrivals (relay everything through; line ends send 0); Ring feeds
//       num_sends = its short-way depth (fwd N/2, bwd (N-1)/2) so each block
//       travels only the short way round and lands EXACTLY once per device.
//   Every fabric write lands DIRECTLY into the downstream device's gather_buffer at
//   the block's canonical page range (uniform mesh buffer address, local accessor
//   routed one hop); one counting inc per landed block tells that device's reader
//   the block arrived (in-order on the connection, after the block's pages). An
//   idle direction (num_sends==0: line ends, ring N==2 backward) opens no
//   connection, reads NO rt args (its rt-arg list is empty — §R10), and returns.
//   The op owns what the helper does NOT: the block order, the block page
//   addressing (gb_page = c*P + p), and the counting atomic-inc target.
//
// PHASE B (reduce, phase==1): write each summed slice tile from cb_summed_slice to
//   its DENSE output page start+t (the output slice is dense by construction; the
//   reader's t-th pushed group corresponds to output page start+t because both
//   derive from the SAME per-core (start, n) rt args — §R6). Pure local NoC writes.
//
// Uniform CT superset keeps the discarded if-constexpr branch in-bounds:
//   [0]=phase, [1..7]=scalars, then ONE TensorAccessorArgs (gather_buffer for
//   gather; output for reduce).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

constexpr uint32_t PHASE_GATHER = 0;
constexpr uint32_t PHASE_REDUCE = 1;

void kernel_main() {
    constexpr uint32_t phase = get_compile_time_arg_val(0);

    if constexpr (phase == PHASE_GATHER) {
        // ---------------------------------------------------------------------
        // Phase A — fabric store-and-forward egress.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_relay_pages = get_compile_time_arg_val(1);
        constexpr uint32_t direction = get_compile_time_arg_val(2);  // 0 = forward, 1 = backward
        constexpr uint32_t my_chip_id = get_compile_time_arg_val(3);
        constexpr uint32_t ring_size = get_compile_time_arg_val(4);
        // Blocks this direction sends (seed + relays), HOST-derived from the topology
        // (see file banner). 0 = idle direction (line ends, ring N==2 backward).
        constexpr uint32_t num_sends = get_compile_time_arg_val(5);
        constexpr uint32_t alignment = get_compile_time_arg_val(6);
        // slot 7 is a zero pad (uniform CT superset).
        constexpr auto gather_buffer_args = TensorAccessorArgs<8>();

        // Idle direction: no fabric egress, no connection opened, no rt args.
        if constexpr (num_sends > 0) {
            uint32_t ai = 0;
            const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
            const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
            const uint32_t page_size = get_arg_val<uint32_t>(ai++);
            const uint32_t num_hops = get_arg_val<uint32_t>(ai++);
            const uint32_t counting_sem_addr = get_arg_val<uint32_t>(ai++);
            const uint32_t target_noc_x = get_arg_val<uint32_t>(ai++);
            const uint32_t target_noc_y = get_arg_val<uint32_t>(ai++);

            // Fabric connection arg block (laid out by append_ccl_fabric_rt_args); its
            // leading has_forward flag also encodes the send direction, which we peek.
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
            // Relay indices are ring-modular; the line values never wrap, so this is
            // exactly the pre-ring `my_chip_id -/+ k` walk there.
            for (uint32_t k = 0; k < num_sends; ++k) {
                uint32_t c;
                if (k == 0) {
                    c = my_chip_id;  // seed
                } else {
                    c = (direction == 0) ? (my_chip_id + ring_size - k) % ring_size
                                         : (my_chip_id + k) % ring_size;  // relays
                }
                for (uint32_t p = 0; p < P; ++p) {
                    cb_wait_front(cb_relay_pages, 1);
                    const uint32_t l1 = get_read_ptr(cb_relay_pages);
                    writer.write_page(l1, c * P + p, gather_buffer);
                    noc_async_writes_flushed();  // §R8: the page must be read before CB slot reuse
                    cb_pop_front(cb_relay_pages, 1);
                }
                // In-order on the connection: this inc lands after the block's data.
                counter.inc(neighbor_sem);
            }

            stream.close();  // drains (write + atomic barriers) then closes
        }
    } else {
        // ---------------------------------------------------------------------
        // Phase B — write each summed slice tile to its dense output page.
        // ---------------------------------------------------------------------
        constexpr uint32_t cb_summed_slice = get_compile_time_arg_val(1);
        constexpr auto output_args = TensorAccessorArgs<8>();

        uint32_t ai = 0;
        const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
        const uint32_t page_size = get_arg_val<uint32_t>(ai++);
        const uint32_t start_tile = get_arg_val<uint32_t>(ai++);  // first owned OUTPUT position
        const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);   // owned OUTPUT positions

        const auto output = TensorAccessor(output_args, output_addr, page_size);

        for (uint32_t t = 0; t < num_tiles; ++t) {
            cb_wait_front(cb_summed_slice, 1);
            const uint32_t l1 = get_read_ptr(cb_summed_slice);
            noc_async_write(l1, output.get_noc_addr(start_tile + t), page_size);
            noc_async_write_barrier();
            cb_pop_front(cb_summed_slice, 1);
        }
    }
}
