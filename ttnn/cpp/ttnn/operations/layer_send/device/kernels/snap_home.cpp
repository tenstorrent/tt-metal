// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _HOME_KERNEL (SnapshotSend static)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row (a page of the ring and of the slab)
constexpr uint32_t NB = 4;                              // blocks
constexpr uint32_t BR = 32;                             // rows a block
constexpr auto g_args = TensorAccessorArgs<1>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr0..rr3 | kind0..3, mask0..3, sem0..3, nports, (px, py) x nports]
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t slab = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // BR rows, 64 B aligned
    const uint32_t zbuf = buf + BR * PAGE;                // one zero row
    bool zeroed = false;
    for (uint32_t b = 0; b < NB; ++b) {
        const uint32_t kind = get_arg_val<uint32_t>(7 + b);
        if (kind == 0) {
            if (!zeroed) {
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbuf);
                for (uint32_t i = 0; i < PAGE / 4; ++i) {
                    z[i] = 0;
                }
                zeroed = true;
            }
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_write(zbuf, S.get_noc_addr(page0 + b * BR + i), PAGE);
            }
        } else if (kind == 1) {
            const uint32_t rr = get_arg_val<uint32_t>(3 + b);
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_read(G.get_noc_addr(rr + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_write(buf + i * PAGE, S.get_noc_addr(page0 + b * BR + i), PAGE);
            }
            noc_async_write_barrier();  // (buf is reused by the next block)
        }
    }
    const uint32_t nports = get_arg_val<uint32_t>(19);
    for (uint32_t b = 0; b < NB; ++b) {
        if (get_arg_val<uint32_t>(7 + b) != 2) {
            continue;
        }
        volatile tt_l1_ptr uint32_t* arrived =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(15 + b));
        noc_semaphore_wait_min(arrived, 1);
        const uint32_t mask = get_arg_val<uint32_t>(11 + b);
        for (uint32_t k = 0; k < nports; ++k) {
            if (mask & (1u << k)) {
                const uint32_t px = get_arg_val<uint32_t>(20 + 2 * k);
                const uint32_t py = get_arg_val<uint32_t>(21 + 2 * k);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(b)), 1);
            }
        }
    }
    noc_async_write_barrier();
    for (uint32_t b = 0; b < NB; ++b) {
        if (get_arg_val<uint32_t>(7 + b) == 2) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(15 + b)), 0u - 1u);
        }
    }
    noc_async_atomic_barrier();
}
