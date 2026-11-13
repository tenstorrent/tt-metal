// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// Waits for all chips' slots in the active ping-pong half, publishes the half index to
// compute via cb_meta, and exposes the active half in cb_reduction.
void kernel_main() {
    constexpr uint32_t cb_reduction_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t group_pages = get_compile_time_arg_val(3);

    const uint32_t sem_even_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_odd_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_iter_addr = get_arg_val<uint32_t>(2);

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* iter_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_iter_addr);
    const uint32_t iter = *iter_ptr;
    *iter_ptr = iter + 1u;

    const uint32_t half = iter & 1u;
    const uint32_t target = (iter / 2u + 1u) * ring_size;
    volatile tt_l1_ptr uint32_t* sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(half ? sem_odd_addr : sem_even_addr);
    noc_semaphore_wait_min(sem_ptr, target);

    CircularBuffer cb_meta(cb_meta_id);
    cb_meta.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_meta.get_write_ptr())[0] = half;
    cb_meta.push_back(1);

    CircularBuffer cb_reduction(cb_reduction_id);
    cb_reduction.push_back((half + 1u) * ring_size * group_pages);
}
