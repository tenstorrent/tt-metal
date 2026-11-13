// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// Streams this link's input shards into cb0 in a few pipelined batches (1, 3, 6, 6...) so
// RISCV_0 starts sending on the first chunk (~0.5 us) while avoiding per-reducer read barriers.
// Note: cb0 holds all link_pages without wrapping in one invocation, so slot addresses remain
// valid for NOC_1 writes after cb0.push_back().
void kernel_main() {
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t group_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_pages = get_compile_time_arg_val(4);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);

    size_t arg_idx = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_even_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_odd_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_iter_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(arg_idx++);

    tt_l1_ptr uint32_t* red_x = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_reducers;
    tt_l1_ptr uint32_t* red_y = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_reducers;

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* iter_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_iter_addr);
    const uint32_t iter = *iter_ptr;
    *iter_ptr = iter + 1u;
    const uint32_t half = iter & 1u;

    Noc noc;
    CircularBuffer cb0(cb0_id);
    const uint32_t dst_slot_addr = scratch_addr + (half * ring_size + my_chip_id) * group_bytes;

    // Issue reads in progressive batches: 1 chunk first to wake writer quickly, then 3, then 6...
    uint32_t r = 0;
    uint32_t batch_size = 1;
    while (r < num_reducers) {
        const uint32_t batch_end = (r + batch_size < num_reducers) ? (r + batch_size) : num_reducers;
        const uint32_t count = batch_end - r;
        const uint32_t batch_pages = count * group_pages;

        cb0.reserve_back(batch_pages);
        const uint32_t batch_base = cb0.get_write_ptr();
        uint32_t write_ptr = batch_base;
        for (uint32_t j = r; j < batch_end; ++j) {
            const uint32_t num_chunks = get_arg_val<uint32_t>(arg_idx++);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t x = get_arg_val<uint32_t>(arg_idx++);
                const uint32_t y = get_arg_val<uint32_t>(arg_idx++);
                const uint32_t off = get_arg_val<uint32_t>(arg_idx++);
                const uint32_t len = get_arg_val<uint32_t>(arg_idx++);
                noc_async_read(get_noc_addr(x, y, input_addr + off), write_ptr, len);
                write_ptr += len;
            }
        }
        noc.async_read_barrier();
        cb0.push_back(batch_pages);

        uint32_t read_ptr = batch_base;
        for (uint32_t j = r; j < batch_end; ++j) {
            noc_async_write(read_ptr, get_noc_addr(red_x[j], red_y[j], dst_slot_addr), group_bytes);
            read_ptr += group_bytes;
        }

        r = batch_end;
        batch_size = (batch_size == 1u) ? 3u : 6u;
    }

    noc.async_write_barrier();
    const uint32_t sem_addr = half ? sem_odd_addr : sem_even_addr;
    for (uint32_t j = 0; j < num_reducers; ++j) {
        noc_semaphore_inc(get_noc_addr(red_x[j], red_y[j], sem_addr), 1);
    }
    noc.async_atomic_barrier();
}
