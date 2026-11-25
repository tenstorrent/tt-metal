// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem transport bench: port reader (RISCV_1 / NoC0). Stands in for q_stem_port_reader.cpp: fills cb_port_stage
// with `total` segments of hw tiles in batches of up to `batch` segments per read barrier (a batch never wraps past
// the CB end, the CB holds 2 batches), as the op does. The source is a ring of n_src segments in this core's own L1
// (the op pulls the same bytes from the stage-1 cores; the pull was measured non-binding in Refinement 3), so the
// bench times the send loop + fabric, not stage 1.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_port_stage = get_compile_time_arg_val(0);
    constexpr uint32_t seg_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t hw = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t cap = 2 * batch;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_src = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);

    uint32_t slot = 0, src_slot = 0, done = 0;
    while (done < total) {
        uint32_t limit = cap - slot < batch ? cap - slot : batch;
        if (limit > total - done) {
            limit = total - done;
        }
        cb_reserve_back(cb_port_stage, limit * hw);
        uint32_t l1 = get_write_ptr(cb_port_stage);
        for (uint32_t i = 0; i < limit; ++i) {
            noc_async_read(get_noc_addr(src_addr + src_slot * seg_bytes), l1, seg_bytes);
            l1 += seg_bytes;
            src_slot = src_slot + 1 == n_src ? 0 : src_slot + 1;
        }
        noc_async_read_barrier();
        cb_push_back(cb_port_stage, limit * hw);
        slot = (slot + limit) % cap;
        done += limit;
    }
}
