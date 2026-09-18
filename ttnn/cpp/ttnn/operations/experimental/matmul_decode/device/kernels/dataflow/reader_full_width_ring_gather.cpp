// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Ring-gather reader for in0. Implements one direction (CW or CCW) of a pipelined
// *closed*-ring all-gather on the activations, mirroring the matmul-1D GATHER_IN0 pattern
// (ttnn/cpp/ttnn/operations/matmul/in0_ring_gather.md) but generalized so:
//   1. The set of sources (S) and the set of compute cores (C) can be disjoint OR
//      overlapping. In either case, the walk is a closed ring over S ∪ C with wraparound
//      from the last core back to the first.
//   2. The number of compute cores can be smaller (or larger) than the number of sources;
//      non-source cores act as pure receiver-forwarders, and non-compute sources
//      participate in relaying peers' shards.
//   3. The same kernel is instantiated twice per compute core (RISCV_1/NOC_0 for CW, and
//      RISCV_0/NOC_1 for CCW) so both NoCs move traffic in parallel.
//
// Ring walk: closed ring over S ∪ C. Each core has a role (SOURCE = owns local shard,
// HOP = pure relay) and per-core (num_recv, num_sends) computed host-side:
//   num_recv(p)  = S - is_source(p)  (every non-own source's shard visits p)
//   num_sends(p) = is_source(p) + num_recv(p) - is_terminator(p)
// where is_terminator(p) = is_source((p+1) % W): if our successor is a source, we're the
// last hop for the shard that would otherwise loop back to it, so we receive but don't
// forward one shard. This saves S hops per ring-launch.
//
// Slot mapping into cb_in2 is temporal: the k-th arriving shard lands in slot (k-1) of
// this core's cb_in2 (indexed from 0 for the first arrival). Both a core X and its
// successor agree on this because cb_in2 has the same L1 offset on every core in the
// ring and X tracks "shards_sent_so_far" locally.
//
// Semaphore accounting: at step t (t >= 1) the receiver blocks until it has been
// bumped `t` times, which guarantees the t-th shard has landed in its cb_in2 before
// it is read for forwarding. The write / atomic ordering matches the reference (flush
// the write buffer before bumping the atomic buffer).

enum : uint32_t {
    ROLE_IDLE = 0,
    ROLE_SOURCE = 1,  // owns a shard traveling in this direction; injects at step 0
    ROLE_HOP = 2,     // no local shard; forwards/receives only (compute-only core in this dir)
};

void kernel_main() {
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t sig_sem_id = get_compile_time_arg_val(2);
    // Signal-in0 is a one-shot cb_in0 push_back so compute can wait_front on a globally-
    // allocated CB. Only one reader per compute core should be tagged with signal_in0=1.
    constexpr uint32_t local_shard_num_tiles = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in2_id = get_named_compile_time_arg_val("cb_in2");

    uint32_t rt_idx = 0;
    const uint32_t role = get_arg_val<uint32_t>(rt_idx++);
    if (role == ROLE_IDLE) {
        return;
    }
    const uint32_t num_recv = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_sends = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t next_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t next_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t signal_in0 = get_arg_val<uint32_t>(rt_idx++);

    const bool has_own = (role == ROLE_SOURCE);
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    Noc noc;
    Semaphore<> sig_sem(sig_sem_id);
    CircularBuffer cb_in2(cb_in2_id);

    // Reserve room for every shard we will receive in this direction up front, exactly like the
    // reference matmul-1D reader. get_write_ptr() then gives us the base L1 offset of cb_in2 on
    // this core, which -- because cb_in2 has identical geometry on every ring member -- is also
    // the base L1 offset on our successor. Slot addressing is done manually with base + slot *
    // shard_size_bytes and push_back is called once per received shard to hand it to the next
    // consumer (either compute or another forward step).
    if (num_recv > 0) {
        cb_in2.reserve_back(num_recv * shard_num_tiles);
    }
    const uint32_t l1_cb_in2_base = cb_in2.get_write_ptr();

    uint32_t shards_sent_so_far = 0;

    // Step 0: sources inject their local shard into successor's cb_in2 slot 0.
    // The last core in the walk has num_sends == 0; a source there simply skips the send
    // (its shard is only used locally on itself if it's also a compute core, otherwise it is
    // an ill-formed configuration -- host should place the last-walk core as a compute core).
    if (has_own && num_sends > 0) {
        CircularBuffer cb_in0(cb_in0_id);
        const uint32_t local_read_addr = cb_in0.get_read_ptr();
        const uint32_t dst_addr = l1_cb_in2_base;  // slot 0

        UnicastEndpoint dst_ep;
        noc.async_write(
            CoreLocalMem<uint32_t>(local_read_addr),
            dst_ep,
            shard_size_bytes,
            {},
            {.noc_x = next_x, .noc_y = next_y, .addr = dst_addr});
        // Flush the payload before bumping the receiver's semaphore: the write and the atomic
        // increment use different NoC command buffers, so without the flush the atomic can land
        // ahead of the payload and the successor may read stale L1.
        noc.async_writes_flushed();
        sig_sem.up(noc, next_x, next_y, 1);
        shards_sent_so_far = 1;
    }

    // We must issue exactly num_sends NoC writes in total (num_recv forwards +
    // has_own injection at t=0, capped at 0 for the last core in the walk). Since forwards must
    // happen strictly after their corresponding receive, and both source-with-own and hop cores
    // walk the same t=1..num_recv receive schedule, "still owe a forward" is simply
    // shards_sent_so_far < num_sends. Using t < num_sends is wrong for hop cores because they
    // don't consume an injection quota at t=0, so the loop stops one short and the downstream
    // wait_min(num_recv) never resolves.
    const uint32_t max_steps = num_recv + 1;  // t = 1..num_recv inclusive

    for (uint32_t t = 1; t < max_steps; t++) {
        const bool receiving = true;  // t is in [1, num_recv]

        if (receiving) {
            sig_sem.wait_min(t);  // the t-th shard has now landed in slot (t-1)
        }

        const bool sending = (shards_sent_so_far < num_sends);
        if (sending) {
            // Forward the shard we just received in this step (or, for a source at t=1 whose
            // upstream is silent -- an ill-formed layout we don't currently produce).
            const uint32_t read_slot = t - 1;
            const uint32_t read_addr = l1_cb_in2_base + read_slot * shard_size_bytes;
            const uint32_t dst_addr = l1_cb_in2_base + shards_sent_so_far * shard_size_bytes;

            UnicastEndpoint dst_ep;
            noc.async_write(
                CoreLocalMem<uint32_t>(read_addr),
                dst_ep,
                shard_size_bytes,
                {},
                {.noc_x = next_x, .noc_y = next_y, .addr = dst_addr});
            noc.async_writes_flushed();
            sig_sem.up(noc, next_x, next_y, 1);
            shards_sent_so_far++;
        }

        if (receiving) {
            cb_in2.push_back(shard_num_tiles);
        }
    }

    // cb_in0 is globally allocated over the input tensor's L1 (no producer normally pushes it),
    // so on compute-and-source cores we manually publish it once so the compute kernel's
    // wait_front returns. Only one reader per core carries signal_in0 == 1 to avoid double push.
    if (signal_in0 && local_shard_num_tiles > 0) {
        CircularBuffer cb_in0(cb_in0_id);
        cb_in0.reserve_back(local_shard_num_tiles);
        cb_in0.push_back(local_shard_num_tiles);
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
