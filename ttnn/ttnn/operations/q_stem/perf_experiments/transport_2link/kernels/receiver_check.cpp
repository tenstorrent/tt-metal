// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem transport bench: destination core with an ORDERING race check (for the flush=false fused increment).
// Like receiver.cpp, but walks the (source, row) increments in host order: wait until source s's counter reaches the
// threshold of a row, then IMMEDIATELY read the last word of the last packet that increment was fused with (the data
// most likely to be overtaken by its increment) and compare it with the expected pattern word. A mismatch = the
// increment became visible before its write's data: counted into err (this core's word of the errs tensor).
// The landing is zeroed by the host before every checked call, so a stale word cannot pass.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    size_t a = 0;
    const uint32_t err_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t srcs = a;
    a += 2 * n;
    const uint32_t n_checks = get_arg_val<uint32_t>(a++);
    uint32_t err = 0, first_bad = 0xFFFFFFFF;
    for (uint32_t c = 0; c < n_checks; ++c) {
        const uint32_t si = get_arg_val<uint32_t>(a++);
        const uint32_t thr = get_arg_val<uint32_t>(a++);
        const uint32_t word = get_arg_val<uint32_t>(a++);
        const uint32_t expect = get_arg_val<uint32_t>(a++);
        volatile tt_l1_ptr uint32_t* ctr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(srcs + 2 * si));
        while (*ctr < thr) {
        }
        const uint32_t got = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word);
        if (got != expect) {
            ++err;
            if (first_bad == 0xFFFFFFFF) {
                first_bad = c;
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t addr = get_arg_val<uint32_t>(srcs + 2 * i);
        const uint32_t expect = get_arg_val<uint32_t>(srcs + 2 * i + 1);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr), expect);
        noc_semaphore_inc(get_noc_addr(addr), 0u - expect);
    }
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(err_addr);
    e[0] = err;
    e[1] = first_bad;
    e[2] = n_checks;
}
