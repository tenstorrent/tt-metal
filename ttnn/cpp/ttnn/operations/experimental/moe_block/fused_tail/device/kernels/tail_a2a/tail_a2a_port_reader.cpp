// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A port reader (RISCV_1, NoC0): my runs of every destination, row tile by row tile, from the staging buffer z
// once its workers wrote the row tile.

#include "tail_a2a_port_common.hpp"

void kernel_main() {
    constexpr auto z_args = TensorAccessorArgs<6>();
    size_t a = 0;
    const uint32_t z_addr = get_arg_val<uint32_t>(a++), NW = get_arg_val<uint32_t>(a++);
    const uint32_t zr_addr = get_arg_val<uint32_t>(a++), RT = get_arg_val<uint32_t>(a++);
    const uint32_t NTH = get_arg_val<uint32_t>(a++), NTB = get_arg_val<uint32_t>(a++);
    const uint32_t l = get_arg_val<uint32_t>(a++), L = get_arg_val<uint32_t>(a++), nd = get_arg_val<uint32_t>(a++);
    const uint32_t d0 = a;
    const auto z = TensorAccessor(z_args, z_addr, PB);
    volatile tt_l1_ptr uint32_t* zr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zr_addr);
    if (nd == 0) {
        return;
    }
    uint32_t batch = 0, wptr = 0;
    for (uint32_t r = 0; r < RT; ++r) {
        noc_semaphore_wait_min(zr + 4 * r, NW);  // (every worker wrote row tile r: its word, 16 B apart)
        for (uint32_t k = 0; k < nd; ++k) {
            const uint32_t e = get_arg_val<uint32_t>(d0 + k), q = e & 0xFF, part = (e >> 8) & 3;
            const uint32_t base = r * NTH + q * NTB;
            for_each_run(NTB, l, L, part, [&](uint32_t m, uint32_t n) {
                if (batch == 0) {
                    cb_reserve_back(CB, RP * GROUP);
                    wptr = get_write_ptr(CB);
                }
                for (uint32_t i = 0; i < n; ++i) {  // (z's pages m, m + NB, ..: one bank, consecutive)
                    noc_async_read(z.get_noc_addr(base + m + i * NB), wptr + (batch * RP + i) * PB, PB);
                }
                if (++batch == GROUP) {
                    noc_async_read_barrier();
                    cb_push_back(CB, RP * GROUP);
                    batch = 0;
                }
            });
        }
    }
    if (batch) {
        noc_async_read_barrier();
        cb_push_back(CB, RP * batch);
    }
    for (uint32_t r = 0; r < RT; ++r) {  // (re-arm: every increment of the call landed)
        noc_semaphore_inc(get_noc_addr(zr_addr + 16 * r), 0u - NW);
    }
    noc_async_atomic_barrier();
}
