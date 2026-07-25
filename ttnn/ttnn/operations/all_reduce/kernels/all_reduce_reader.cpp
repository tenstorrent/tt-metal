// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — reader (NCRISC). Three sequential phases, no compute.
//
//   Phase 1  Stage this device's P input pages into cb_broadcast_pages, which the
//            writer drains onto the fabric (duplex multicast to every peer).
//
//   Phase 2  Barrier on arrivals: noc_semaphore_wait_min(sem, N-1). Each peer's
//            LAST multicast packet is a FUSED write+atomic-inc with flush=true, so
//            exactly N-1 increments arrive and `sem >= N-1` implies every peer's
//            whole shard has landed in gathered DRAM. Then re-arm the counter
//            (noc_semaphore_set(sem, 0)) — programs are cached and the
//            GlobalSemaphore is reused, and a RECEIVER resets AFTER its wait.
//            The WAITING half of a cross-device sync, the receive INGRESS (a local
//            noc_async_read; there is no FabricStreamReceiver) and the re-arm are
//            all op-owned; the CCL dataflow helper deliberately does not wrap them.
//
//   Phase 3  For each output tile p, gather the N contributions into ONE block of
//            N CONTIGUOUS pages of cb_shard_tiles, ordered device 0..N-1:
//              k == my_chip_id -> input_tensor page p   (the local shard is read
//                                 straight from the input, so slot my_id of the
//                                 gathered buffer is never written and needs no
//                                 extra writer->reader handshake)
//              k != my_chip_id -> gathered_tensor page k*P + p
//            cb_shard_tiles holds 2*N pages and every push/pop is exactly N, so
//            the write pointer is always at page offset 0 or N: the N pages a
//            single cb_reserve_back yields are contiguous and base + k*page_size
//            never wraps.
//
// Ordering: phase 1 completes before phase 2 begins, so all P pages are already
// pushed when the reader parks on the semaphore — the writer can always finish its
// fabric egress. Reader and writer are different RISCs, and the fabric routers are
// separate ERISC cores that land a peer's payload in DRAM independently of that
// peer's worker, so there is no circular wait.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_broadcast_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_shard_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr auto gathered_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    static_assert(num_devices >= 2, "all_reduce needs at least 2 devices on the line");

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t gathered_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(ai++);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gathered = TensorAccessor(gathered_args, gathered_addr, page_size);
    const uint32_t P = pages_per_shard;
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);

    // ---- Phase 1: stage the local shard for the writer's fabric broadcast ----
    for (uint32_t p = 0; p < P; ++p) {
        cb_reserve_back(cb_broadcast_pages, 1);
        const uint32_t l1 = get_write_ptr(cb_broadcast_pages);
        noc_async_read(input.get_noc_addr(p), l1, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_broadcast_pages, 1);
    }

    // ---- Phase 2: barrier on the N-1 peer arrivals, then re-arm the counter ----
    noc_semaphore_wait_min(sem_ptr, num_devices - 1);
    noc_semaphore_set(sem_ptr, 0);

    // ---- Phase 3: interleave the N contributions to each output tile ----
    for (uint32_t p = 0; p < P; ++p) {
        cb_reserve_back(cb_shard_tiles, num_devices);
        const uint32_t base = get_write_ptr(cb_shard_tiles);
        for (uint32_t k = 0; k < num_devices; ++k) {
            const uint64_t src = (k == my_chip_id) ? input.get_noc_addr(p) : gathered.get_noc_addr(k * P + p);
            noc_async_read(src, base + k * page_size, page_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_shard_tiles, num_devices);
    }
}
