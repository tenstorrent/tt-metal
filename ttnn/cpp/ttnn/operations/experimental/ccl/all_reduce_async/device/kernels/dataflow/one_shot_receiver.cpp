// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_reduce_async one_shot: output cores. See create_one_shot_descriptor in the program factory.
//   owner  (this device owns the core's slice): copy the local partial into reduction-buffer slot my_chip, wait for
//          the ring_size - 1 remote reduce-scatter increments, hand the ring_size slots to the reduction kernel (the
//          default path's operand layout), and once the reduced chunk is packed into the output shard, tell this
//          core's link senders (one per direction) that they can multicast it.
//   gather (a peer owns the slice): wait for the owner's all-gather increment (its chunk has landed in the output).
// Both reset the caller's global semaphore to 0 afterwards, as the default path does.

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include <cstdint>

constexpr uint32_t reduction_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
constexpr bool fuse_residual = get_compile_time_arg_val(2) != 0;
constexpr uint32_t residual_cb_id = get_compile_time_arg_val(3);
constexpr uint32_t num_residual_tiles = get_compile_time_arg_val(4);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(5);
constexpr uint32_t in_page_size = get_compile_time_arg_val(6);
constexpr uint32_t ring_size = get_compile_time_arg_val(7);
constexpr uint32_t ag_ready_sem_id = get_compile_time_arg_val(8);

constexpr uint32_t kRoleNone = 0;
constexpr uint32_t kRoleOwner = 2;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t role = get_arg_val<uint32_t>(arg_idx++);
    if (role == kRoleNone) {
        return;
    }
    const uint32_t wait_value = get_arg_val<uint32_t>(arg_idx++);
    // Device 2.0 migration: legacy primitive retained: the semaphore is a GlobalSemaphore address.
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_common_arg_val<uint32_t>(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore));

    if (role != kRoleOwner) {
        noc_semaphore_wait(sem, wait_value);
        noc_semaphore_set(sem, 0);
        return;
    }

    const uint32_t my_chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_senders = get_arg_val<uint32_t>(arg_idx++);
    const size_t sender_arg_base = arg_idx;  // per sender of this core's link: (noc x << 16 | y)
    arg_idx += num_senders;
    const uint32_t num_segs = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t in_addr = get_common_arg_val<uint32_t>(1);

    if constexpr (fuse_residual) {
        // The residual shard is resident in L1 (CB backed by the residual tensor): just publish it.
        CircularBuffer cb_res(residual_cb_id);
        cb_res.push_back(num_residual_tiles);
    }

    // Local contribution -> slot my_chip (remote devices fill the other slots).
    uint32_t l1_write_addr = get_write_ptr(reduction_cb_id) + my_chip * chunk_tiles * in_page_size;
    for (uint32_t s = 0; s < num_segs; s++) {
        const uint32_t xy = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t first_tile = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t tiles = get_arg_val<uint32_t>(arg_idx++);
        noc_async_read(
            get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + first_tile * in_page_size),
            l1_write_addr,
            tiles * in_page_size);
        l1_write_addr += tiles * in_page_size;
    }
    noc_async_read_barrier();

    noc_semaphore_wait(sem, wait_value);
    noc_semaphore_set(sem, 0);

    CircularBuffer cb(reduction_cb_id);
    cb.push_back(ring_size * chunk_tiles);

    // Reduced chunk packed into this core's output shard -> this link's senders may multicast it.
    CircularBuffer cb_out(out_cb_id);
    cb_out.wait_front(chunk_tiles);
    for (uint32_t i = 0; i < num_senders; i++) {
        const uint32_t sender_xy = get_arg_val<uint32_t>(sender_arg_base + i);
        noc_semaphore_inc(get_noc_addr(sender_xy >> 16, sender_xy & 0xFFFF, get_semaphore(ag_ready_sem_id)), 1);
    }
    noc_async_atomic_barrier();
}
