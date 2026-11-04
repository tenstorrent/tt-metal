// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include <cstdint>

// The fused all-reduce's reader on one fabric sender core: all_reduce_async's worker_reader.cpp, except
// that the slots it forwards are written by this same program, so it first waits for each of its
// link's group-0 down cores to bump sem_ar_ready once.
void kernel_main() {
    constexpr uint32_t cb_src_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(2);

    size_t arg_idx = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slot_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pages_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pages_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));

    Semaphore<> ar_ready(sem_ar_ready_id);
    ar_ready.wait(num_cores);
    ar_ready.set(0);

    Noc noc;
    CircularBuffer cb_src(cb_src_id);
    UnicastEndpoint src_ep;
    const uint32_t slot_addr = scratch_addr + slot_offset;
    uint32_t pages_read = 0;
    for (uint32_t core = 0; pages_read < pages_to_read; ++core) {
        const uint32_t n = pages_per_core < pages_to_read - pages_read ? pages_per_core : pages_to_read - pages_read;
        cb_src.reserve_back(n);
        noc.async_read(
            src_ep,
            cb_src,
            n * page_size,
            {.noc_x = core_noc_x[core], .noc_y = core_noc_y[core], .addr = slot_addr},
            {});
        noc.async_read_barrier();
        cb_src.push_back(n);
        pages_read += n;
    }
}
