// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Ring-gather reader for in0 in partial-width-sharded matmul_decode. Same *closed*-ring
// pipeline as reader_full_width_ring_gather.cpp, but the destination on every hop is the
// receiver's cb_full_in0 slot indexed by *sender_id* (not by arrival order). This matches
// compute_partial_width_sharded.cpp's `in0_tile = sender * sender_slice_tiles + kc_local`
// addressing, so the existing compute kernel is reused unchanged: it just wait_front's on
// cb_full_in0 as before.
//
// Because slots are indexed by sender_id (globally unique per shard), every hop writes to
// the same L1 offset on the successor (cb_full_in0 has identical geometry on every ring
// member). Semaphore accounting is per-arrival: at step t the receiver blocks on
// sig_sem >= t, at which point the arriving shard has landed at slot arriving_ids[t-1] and
// can be forwarded to the successor at the same slot.
//
// Own-shard handling for a source-and-compute (overlap) core: cb_full_in0[own_sender_id]
// must contain this core's own K-slice, but the ring never delivers it here (we skip the
// terminator hop that would loop it back to the origin). Fill it in-place with a local L1
// memcpy from cb_in0 before publishing cb_full_in0 to compute. Source-only cores do the
// injection to the successor but don't need the local copy (no compute here).

enum : uint32_t {
    RG_ROLE_IDLE = 0,
    RG_ROLE_SOURCE_ONLY = 1,         // has own shard, not a compute core here
    RG_ROLE_HOP = 2,                 // no own shard, compute core (or pure relay)
    RG_ROLE_SOURCE_AND_COMPUTE = 3,  // has own shard AND is a compute core
};

void kernel_main() {
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t sig_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t full_in0_num_tiles = get_compile_time_arg_val(3);
    // cb_in1 is aliased to the L1-resident weight tensor and has no producer of its own; this
    // reader publishes it on compute cores so compute's wait_front(cb_in1) resolves. Mirrors
    // the equivalent one-shot push in reader_partial_width_sharded.cpp's non-GCB branch.
    constexpr uint32_t in1_slab_num_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t cb_in0_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_full_in0_id = get_named_compile_time_arg_val("cb_full_in0");
    constexpr uint32_t cb_in1_id = get_named_compile_time_arg_val("cb_in1");

    uint32_t rt_idx = 0;
    const uint32_t role = get_arg_val<uint32_t>(rt_idx++);
    if (role == RG_ROLE_IDLE) {
        return;
    }
    const uint32_t num_recv = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_sends = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t next_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t next_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t own_sender_id = get_arg_val<uint32_t>(rt_idx++);
    volatile tt_l1_ptr uint32_t* arriving_ids = nullptr;
    if (num_recv > 0) {
        arriving_ids = (volatile tt_l1_ptr uint32_t*)get_arg_addr(rt_idx);
        rt_idx += num_recv;
    }

    const bool has_own = (role == RG_ROLE_SOURCE_ONLY) || (role == RG_ROLE_SOURCE_AND_COMPUTE);
    const bool is_compute = (role == RG_ROLE_HOP) || (role == RG_ROLE_SOURCE_AND_COMPUTE);
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    Noc noc;
    Semaphore<> sig_sem(sig_sem_id);
    CircularBuffer cb_full_in0(cb_full_in0_id);

    // Reserve the whole cb_full_in0 up front so get_write_ptr returns the stable base of the
    // single big slot. Every write into cb_full_in0 (local memcpy + remote ring writes) is
    // done at an absolute offset base + sender_id * shard_size_bytes; the push_back at the
    // end atomically hands the fully-assembled tensor to compute's wait_front.
    cb_full_in0.reserve_back(full_in0_num_tiles);
    const uint32_t l1_full_in0_base = cb_full_in0.get_write_ptr();

    uint32_t shards_sent_so_far = 0;

    // Overlap core: pre-populate own slot from cb_in0. The ring never delivers the origin's
    // own shard back to itself (terminator saves that hop), so compute would otherwise read
    // stale L1 for own_sender_id's slot. A short uint32 loop is enough -- shard_size_bytes
    // is O(a few tens of KB) and this is off the NoC critical path.
    if (has_own && is_compute) {
        CircularBuffer cb_in0(cb_in0_id);
        const uint32_t src = cb_in0.get_read_ptr();
        const uint32_t dst = l1_full_in0_base + own_sender_id * shard_size_bytes;
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        const uint32_t words = shard_size_bytes >> 2;
        for (uint32_t i = 0; i < words; ++i) {
            d[i] = s[i];
        }
    }

    // Step 0: sources inject their own shard into successor's cb_full_in0[own_sender_id]. The
    // last core in the walk has num_sends == 0 (its successor is a source and this shard would
    // just be looped back); a source there skips the send.
    if (has_own && num_sends > 0) {
        CircularBuffer cb_in0(cb_in0_id);
        const uint32_t local_read_addr = cb_in0.get_read_ptr();
        const uint32_t dst_addr = l1_full_in0_base + own_sender_id * shard_size_bytes;

        UnicastEndpoint dst_ep;
        noc.async_write(
            CoreLocalMem<uint32_t>(local_read_addr),
            dst_ep,
            shard_size_bytes,
            {},
            {.noc_x = next_x, .noc_y = next_y, .addr = dst_addr});
        noc.async_writes_flushed();
        sig_sem.up(noc, next_x, next_y, 1);
        shards_sent_so_far = 1;
    }

    // Forward received shards along the ring. At step t (t = 1..num_recv) the t-th shard has
    // landed at our cb_full_in0[arriving_ids[t-1]]; if we still owe a forward (source and hop
    // cores share the same num_recv schedule), copy it to the successor at the same slot.
    // Using shards_sent_so_far < num_sends (rather than t < num_sends) is essential for hop
    // cores -- they don't consume a quota at t=0, so t-indexed cutoffs would fall one short.
    const uint32_t max_steps = num_recv + 1;
    for (uint32_t t = 1; t < max_steps; t++) {
        sig_sem.wait_min(t);

        if (shards_sent_so_far < num_sends) {
            const uint32_t sender_id = arriving_ids[t - 1];
            const uint32_t slot_addr = l1_full_in0_base + sender_id * shard_size_bytes;

            UnicastEndpoint dst_ep;
            noc.async_write(
                CoreLocalMem<uint32_t>(slot_addr),
                dst_ep,
                shard_size_bytes,
                {},
                {.noc_x = next_x, .noc_y = next_y, .addr = slot_addr});
            noc.async_writes_flushed();
            sig_sem.up(noc, next_x, next_y, 1);
            shards_sent_so_far++;
        }
    }

    // Publish cb_full_in0 to compute. Only compute cores actually consume it; source-only
    // cores skip the push_back (nothing waits on it here).
    if (is_compute) {
        cb_full_in0.push_back(full_in0_num_tiles);
        // Publish the L1-resident weight so compute's wait_front(cb_in1) returns. cb_in1 is
        // aliased to the input B tensor (no producer of its own); this is a one-shot logical
        // "these tiles are already here" push, symmetric with reader_partial_width_sharded.cpp.
        CircularBuffer cb_in1(cb_in1_id);
        cb_in1.reserve_back(in1_slab_num_tiles);
        cb_in1.push_back(in1_slab_num_tiles);
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
