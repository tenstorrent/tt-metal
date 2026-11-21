// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t MAXB = get_compile_time_arg_val(1);
    constexpr uint32_t BS = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr uint32_t cin = get_compile_time_arg_val(4);
    constexpr uint32_t cmeta = get_compile_time_arg_val(5);
    constexpr uint32_t cscr = get_compile_time_arg_val(6);
    constexpr uint32_t PAGE = get_compile_time_arg_val(7);
    constexpr uint32_t TILE = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto cache_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr uint32_t NG = R == 1 ? 1 : 2;
    constexpr uint32_t TPB = BS / 32;
    constexpr uint32_t RAW_OFF = 128;
    constexpr uint32_t RAW_BYTES = 32;
    constexpr uint32_t RAW_STRIDE = 64;
    constexpr uint32_t PT_OFF = 1024;
    const auto src = TensorAccessor(src_args, get_arg_val<uint32_t>(0));
    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(1));
    const auto ptg = TensorAccessor(pt_args, get_arg_val<uint32_t>(2));
    const uint32_t h = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    cb_reserve_back(cmeta, 1);
    const uint32_t mp = (get_write_ptr(cmeta) + 63) & ~63u;
    for (uint32_t j = 0; j < R; ++j) {
        const auto rp = TensorAccessor(pos_args, get_arg_val<uint32_t>(5 + j));
        noc_async_read(rp.get_noc_addr(0), mp + RAW_OFF + j * RAW_STRIDE, RAW_BYTES);
    }
    noc_async_read(ptg.get_noc_addr(0), mp + PT_OFF, MAXB * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* posv = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mp);
    for (uint32_t j = 0; j < R; ++j) {
        posv[j] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mp + RAW_OFF + j * RAW_STRIDE)[j];
    }
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
    cb_reserve_back(cscr, 1);
    if (first >= 0) {
        volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp + PT_OFF);
        const uint32_t scr = (get_write_ptr(cscr) + 63) & ~63u;
        const uint32_t P0 = static_cast<uint32_t>(first), PL = static_cast<uint32_t>(last);
        const uint32_t t0 = P0 / 32, ng = PL / 32 - t0 + 1;
        for (uint32_t g = 0; g < ng; ++g) {
            const uint32_t pa = (t0 + g) * 32;
            const uint32_t tid = pt[pa / BS] * (NKV * TPB * 4) + h * (TPB * 4) + ((pa % BS) / 32) * 4 + c0;
            const uint32_t lo = (P0 > pa ? P0 : pa) - pa;
            const uint32_t hi = (PL < pa + 31 ? PL : pa + 31) - pa;
            const uint32_t s = scr + g * PAGE;
            noc_async_read(cache.get_noc_addr(tid, 0), s, 64);
            for (uint32_t half = lo / 16; half <= hi / 16; ++half) {
                const uint32_t a0 = half == lo / 16 ? lo % 16 : 0;
                const uint32_t b0 = half == hi / 16 ? hi % 16 : 15;
                const uint32_t off = (a0 / 4) * 64, len = (b0 / 4 + 1) * 64 - off;
                for (uint32_t f = half * 2; f < half * 2 + 2; ++f) {
                    const uint32_t o = 64 + f * 256 + off;
                    noc_async_read(cache.get_noc_addr(tid, o), s + o, len);
                }
            }
        }
    }
    cb_reserve_back(cin, NG);
    const uint32_t dst0 = get_write_ptr(cin);
    if (first >= 0) {
        const uint32_t t0 = static_cast<uint32_t>(first) / 32;
        for (uint32_t j = 0; j < R; ++j) {
            if (posv[j] < 0 || posv[j] != base + static_cast<int32_t>(j)) {
                continue;
            }
            const uint32_t p = static_cast<uint32_t>(posv[j]);
            const uint32_t g = p / 32 - t0;
            const uint32_t r = p % 32;
            const uint32_t fo = r < 16 ? r * 32 : 1024 + (r - 16) * 32;
            const uint64_t a = src.get_noc_addr(j * 4 + c0, h * 32);
            const uint32_t d = dst0 + g * TILE + fo;
            noc_async_read(a, d, 32);
            noc_async_read(a + 512, d + 512, 32);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cscr, 1);
    cb_push_back(cmeta, 1);
    cb_push_back(cin, NG);
}
