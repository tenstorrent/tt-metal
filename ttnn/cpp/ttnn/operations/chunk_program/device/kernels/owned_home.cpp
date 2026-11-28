// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/owned_gather.py's _HOME (after _COMMON), verbatim (NC.7).
#include "owned_common.hpp"

void kernel_main() {
    // [out, me, started, arrived, nports, (px, py) x nports]
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint32_t me = get_arg_val<uint32_t>(1);
    const uint32_t started = get_arg_val<uint32_t>(2);
    const uint32_t arrived = get_arg_val<uint32_t>(3);
    const uint32_t np = get_arg_val<uint32_t>(4);
    uint32_t ndest = 0;
    for (uint32_t d = 0; d < N; ++d) {
        if (d != me && in_group(d, me) && seg_of(d) != 0xFu) {
            ++ndest;
        }
    }
    if (ndest) {  // every other destination of the group runs this program: its buffer is free to write
        wait_min(started, ndest);
        take_back(started, ndest);
    }
    for (uint32_t p = 0; p < np; ++p) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(5 + 2 * p), get_arg_val<uint32_t>(6 + 2 * p), get_semaphore(0)), 1);
    }
    noc_async_atomic_barrier();
    const uint32_t k = seg_of(me);
    if (k == 0xFu) {
        return;
    }
    const uint32_t lo = get_common_arg_val<uint32_t>(NSEG + k);
    const uint32_t hi = get_common_arg_val<uint32_t>(2 * NSEG + k);
    const auto O = TensorAccessor(o_args, out, PAGE);
    const auto S = TensorAccessor(l_args, get_common_arg_val<uint32_t>(k), PAGE);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    if (hi > lo) {  // the own rows: local runs (read, then write; one run at a time per slot)
        const uint32_t nb = banks_of(O);
        const uint32_t base = owner(me) * R;
        const uint32_t total = n_chunks(lo, hi, nb);
        for (uint32_t c0 = 0; c0 < total; c0 += B) {
            const uint32_t m = total - c0 < B ? total - c0 : B;
            uint32_t first, cnt;
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(c0 + i, lo, hi, nb, first, cnt);
                noc_async_read(S.get_noc_addr(first), buf + i * K * PAGE, cnt * PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(c0 + i, lo, hi, nb, first, cnt);
                noc_async_write(buf + i * K * PAGE, O.get_noc_addr(base + first), cnt * PAGE);
            }
            noc_async_write_barrier();
        }
    }
    if (GROUP > 1) {  // every other chip of the group: one increment a link
        wait_min(arrived, (GROUP - 1) * L);
        take_back(arrived, (GROUP - 1) * L);
    }
}
