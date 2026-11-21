// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t BS = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr uint32_t cpk = get_compile_time_arg_val(3);
    constexpr uint32_t cmeta = get_compile_time_arg_val(4);
    constexpr uint32_t cscr = get_compile_time_arg_val(5);
    constexpr uint32_t PAGE = get_compile_time_arg_val(6);
    constexpr auto cache_args = TensorAccessorArgs<7>();
    constexpr uint32_t NG = R == 1 ? 1 : 2;
    constexpr uint32_t TPB = BS / 32;
    constexpr uint32_t PT_OFF = 1024;
    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(0));
    const uint32_t h = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    cb_wait_front(cmeta, 1);
    const uint32_t mp = (get_read_ptr(cmeta) + 63) & ~63u;
    volatile tt_l1_ptr int32_t* posv = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mp);
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp + PT_OFF);
    int32_t first = -1, last = -1, base = 0;
    for (uint32_t j = 0; j < R; ++j) {
        if (posv[j] >= 0) {
            if (first < 0) {
                first = posv[j];
                base = posv[j] - static_cast<int32_t>(j);
            }
            last = posv[j];
        }
    }
    cb_wait_front(cpk, NG);
    cb_wait_front(cscr, 1);
    if (first >= 0) {
        const uint32_t P0 = static_cast<uint32_t>(first), PL = static_cast<uint32_t>(last);
        const uint32_t t0 = P0 / 32, ng = PL / 32 - t0 + 1;
        const uint32_t pk = get_read_ptr(cpk);
        const uint32_t scr = (get_read_ptr(cscr) + 63) & ~63u;
        for (uint32_t g = 0; g < ng; ++g) {
            const uint32_t pa = (t0 + g) * 32;
            const uint32_t tid = pt[pa / BS] * (NKV * TPB * 4) + h * (TPB * 4) + ((pa % BS) / 32) * 4 + c0;
            const uint32_t lo = (P0 > pa ? P0 : pa) - pa;
            const uint32_t hi = (PL < pa + 31 ? PL : pa + 31) - pa;
            const uint32_t s = scr + g * PAGE;
            volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(s);
            volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(pk + g * PAGE);
            for (uint32_t r = lo; r <= hi; ++r) {
                const int32_t slot = static_cast<int32_t>(pa + r) - base;
                if (slot < 0 || slot >= static_cast<int32_t>(R) || posv[slot] != static_cast<int32_t>(pa + r)) {
                    continue;
                }
                const uint32_t fl = (r / 16) * 2, rr = r % 16;
                for (uint32_t f = fl; f < fl + 2; ++f) {
                    d8[f * 16 + rr] = s8[f * 16 + rr];
                    volatile tt_l1_ptr uint32_t* dm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + 64 + f * 256 + rr * 16);
                    volatile tt_l1_ptr uint32_t* sm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pk + g * PAGE + 64 + f * 256 + rr * 16);
                    for (uint32_t w = 0; w < 4; ++w) {
                        dm[w] = sm[w];
                    }
                }
            }
            noc_async_write(s, cache.get_noc_addr(tid, 0), 64);
            for (uint32_t half = lo / 16; half <= hi / 16; ++half) {
                const uint32_t a0 = half == lo / 16 ? lo % 16 : 0;
                const uint32_t b0 = half == hi / 16 ? hi % 16 : 15;
                const uint32_t off = (a0 / 4) * 64, len = (b0 / 4 + 1) * 64 - off;
                for (uint32_t f = half * 2; f < half * 2 + 2; ++f) {
                    const uint32_t o = 64 + f * 256 + off;
                    noc_async_write(s + o, cache.get_noc_addr(tid, o), len);
                }
            }
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cscr, 1);
    cb_pop_front(cpk, NG);
    cb_pop_front(cmeta, 1);
}
